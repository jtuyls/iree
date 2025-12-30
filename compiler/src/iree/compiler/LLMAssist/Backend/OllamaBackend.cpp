// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Backend/OllamaBackend.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <regex>
#include <sstream>
#include <string>

// Platform-specific socket includes
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
#define INVALID_SOCK INVALID_SOCKET
#define CLOSE_SOCKET closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
#define INVALID_SOCK (-1)
#define CLOSE_SOCKET close
#endif

namespace mlir::iree_compiler::LLMAssist {

namespace {

/// Parse a URL into host, port, and path components.
/// Supports format: http://host:port/path or http://host/path
struct ParsedUrl {
  std::string host;
  int port = 80;
  std::string path = "/";
  bool valid = false;

  static ParsedUrl parse(llvm::StringRef url) {
    ParsedUrl result;

    // Remove http:// prefix if present
    if (url.starts_with("http://")) {
      url = url.drop_front(7);
    } else if (url.starts_with("https://")) {
      // HTTPS not supported yet
      return result;
    }

    // Split host:port from path
    auto slashPos = url.find('/');
    llvm::StringRef hostPort = url;
    if (slashPos != llvm::StringRef::npos) {
      hostPort = url.substr(0, slashPos);
      result.path = url.substr(slashPos).str();
    }

    // Split host from port
    auto colonPos = hostPort.find(':');
    if (colonPos != llvm::StringRef::npos) {
      result.host = hostPort.substr(0, colonPos).str();
      llvm::StringRef portStr = hostPort.substr(colonPos + 1);
      if (portStr.getAsInteger(10, result.port)) {
        return result; // Invalid port
      }
    } else {
      result.host = hostPort.str();
    }

    result.valid = !result.host.empty();
    return result;
  }
};

/// Simple HTTP response parser.
struct HttpResponse {
  int statusCode = 0;
  std::string body;
  bool valid = false;

  static HttpResponse parse(const std::string &raw) {
    HttpResponse result;

    // Find the status line
    auto headerEnd = raw.find("\r\n\r\n");
    if (headerEnd == std::string::npos) {
      return result;
    }

    // Parse status code from first line: "HTTP/1.1 200 OK"
    auto firstLineEnd = raw.find("\r\n");
    if (firstLineEnd == std::string::npos) {
      return result;
    }

    std::string statusLine = raw.substr(0, firstLineEnd);
    std::regex statusRegex("HTTP/\\d\\.\\d\\s+(\\d+)");
    std::smatch match;
    if (std::regex_search(statusLine, match, statusRegex) && match.size() > 1) {
      result.statusCode = std::stoi(match[1].str());
    }

    // Extract body
    result.body = raw.substr(headerEnd + 4);

    // Handle chunked transfer encoding by finding the actual JSON
    // Ollama typically sends: size\r\n{json}\r\n0\r\n\r\n
    if (result.body.find('{') != std::string::npos) {
      auto jsonStart = result.body.find('{');
      auto jsonEnd = result.body.rfind('}');
      if (jsonStart != std::string::npos && jsonEnd != std::string::npos) {
        result.body = result.body.substr(jsonStart, jsonEnd - jsonStart + 1);
      }
    }

    result.valid = result.statusCode > 0;
    return result;
  }
};

/// Perform a simple HTTP POST request.
llvm::Expected<HttpResponse> httpPost(const ParsedUrl &url,
                                      const std::string &body,
                                      int timeoutSeconds = 300) {
  // Create socket
  socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == INVALID_SOCK) {
    return llvm::createStringError("Failed to create socket");
  }

  // Set socket timeout
  struct timeval tv;
  tv.tv_sec = timeoutSeconds;
  tv.tv_usec = 0;
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof(tv));

  // Resolve hostname
  struct hostent *host = gethostbyname(url.host.c_str());
  if (!host) {
    CLOSE_SOCKET(sock);
    return llvm::createStringError("Failed to resolve host: %s",
                                   url.host.c_str());
  }

  // Connect
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(url.port);
  addr.sin_addr = *((struct in_addr *)host->h_addr);

  if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    CLOSE_SOCKET(sock);
    return llvm::createStringError("Failed to connect to %s:%d",
                                   url.host.c_str(), url.port);
  }

  // Build HTTP request
  std::ostringstream request;
  request << "POST " << url.path << " HTTP/1.1\r\n";
  request << "Host: " << url.host << ":" << url.port << "\r\n";
  request << "Content-Type: application/json\r\n";
  request << "Content-Length: " << body.size() << "\r\n";
  request << "Connection: close\r\n";
  request << "\r\n";
  request << body;

  std::string requestStr = request.str();

  // Send request
  ssize_t sent = send(sock, requestStr.c_str(), requestStr.size(), 0);
  if (sent < 0 || static_cast<size_t>(sent) != requestStr.size()) {
    CLOSE_SOCKET(sock);
    return llvm::createStringError("Failed to send HTTP request");
  }

  // Receive response
  std::string response;
  char buffer[4096];
  ssize_t received;
  while ((received = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
    buffer[received] = '\0';
    response += buffer;
  }

  CLOSE_SOCKET(sock);

  if (response.empty()) {
    return llvm::createStringError("Empty response from server");
  }

  return HttpResponse::parse(response);
}

/// Perform a simple HTTP GET request (for health checks).
llvm::Expected<HttpResponse> httpGet(const ParsedUrl &url,
                                     int timeoutSeconds = 5) {
  // Create socket
  socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == INVALID_SOCK) {
    return llvm::createStringError("Failed to create socket");
  }

  // Set socket timeout
  struct timeval tv;
  tv.tv_sec = timeoutSeconds;
  tv.tv_usec = 0;
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof(tv));

  // Resolve hostname
  struct hostent *host = gethostbyname(url.host.c_str());
  if (!host) {
    CLOSE_SOCKET(sock);
    return llvm::createStringError("Failed to resolve host: %s",
                                   url.host.c_str());
  }

  // Connect
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(url.port);
  addr.sin_addr = *((struct in_addr *)host->h_addr);

  if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    CLOSE_SOCKET(sock);
    return llvm::createStringError("Failed to connect to %s:%d",
                                   url.host.c_str(), url.port);
  }

  // Build HTTP request
  std::ostringstream request;
  request << "GET " << url.path << " HTTP/1.1\r\n";
  request << "Host: " << url.host << ":" << url.port << "\r\n";
  request << "Connection: close\r\n";
  request << "\r\n";

  std::string requestStr = request.str();

  // Send request
  ssize_t sent = send(sock, requestStr.c_str(), requestStr.size(), 0);
  if (sent < 0 || static_cast<size_t>(sent) != requestStr.size()) {
    CLOSE_SOCKET(sock);
    return llvm::createStringError("Failed to send HTTP request");
  }

  // Receive response
  std::string response;
  char buffer[4096];
  ssize_t received;
  while ((received = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
    buffer[received] = '\0';
    response += buffer;
  }

  CLOSE_SOCKET(sock);

  if (response.empty()) {
    return llvm::createStringError("Empty response from server");
  }

  return HttpResponse::parse(response);
}

} // namespace

OllamaBackend::OllamaBackend(llvm::StringRef endpoint)
    : endpoint_(endpoint.str()) {}

OllamaBackend::~OllamaBackend() = default;

bool OllamaBackend::isAvailable() const {
  // Parse the endpoint URL
  auto url = ParsedUrl::parse(endpoint_);
  if (!url.valid) {
    llvm::errs() << "LLMAssist: Invalid Ollama endpoint URL: " << endpoint_
                 << "\n";
    return false;
  }

  // Check /api/tags endpoint
  url.path = "/api/tags";
  auto response = httpGet(url, 5);
  if (!response) {
    llvm::consumeError(response.takeError());
    return false;
  }

  return response->statusCode == 200;
}

llvm::Expected<GenerationResult>
OllamaBackend::generate(llvm::StringRef prompt,
                        const GenerationConfig &config) {
  auto startTime = std::chrono::steady_clock::now();

  // Parse the endpoint URL
  auto url = ParsedUrl::parse(endpoint_);
  if (!url.valid) {
    return llvm::createStringError("Invalid Ollama endpoint URL: %s",
                                   endpoint_.c_str());
  }
  url.path = "/api/generate";

  // Build JSON request
  llvm::json::Object options;
  options["temperature"] = config.temperature;
  options["num_predict"] = config.maxTokens;
  if (config.seed) {
    options["seed"] = *config.seed;
  }

  llvm::json::Object request;
  request["model"] = config.model;
  request["prompt"] = prompt.str();
  request["stream"] = false;
  request["options"] = std::move(options);

  std::string requestBody;
  llvm::raw_string_ostream os(requestBody);
  os << llvm::json::Value(std::move(request));
  os.flush();

  // Send POST request
  auto response = httpPost(url, requestBody, 300); // 5 minute timeout
  if (!response) {
    return response.takeError();
  }

  if (response->statusCode != 200) {
    return llvm::createStringError("Ollama API returned status %d: %s",
                                   response->statusCode,
                                   response->body.c_str());
  }

  // Parse JSON response
  auto parsed = llvm::json::parse(response->body);
  if (!parsed) {
    std::string errMsg = llvm::toString(parsed.takeError());
    return llvm::createStringError(
        "Failed to parse Ollama response: %s\nRaw response: %s", errMsg.c_str(),
        response->body.c_str());
  }

  auto *obj = parsed->getAsObject();
  if (!obj) {
    return llvm::createStringError("Ollama response is not a JSON object");
  }

  GenerationResult result;

  // Extract response content
  if (auto responseStr = obj->getString("response")) {
    result.content = responseStr->str();
  } else {
    return llvm::createStringError("Ollama response missing 'response' field");
  }

  // Extract token counts (optional)
  if (auto promptTokens = obj->getInteger("prompt_eval_count")) {
    result.promptTokens = *promptTokens;
  }
  if (auto completionTokens = obj->getInteger("eval_count")) {
    result.completionTokens = *completionTokens;
  }

  // Calculate latency
  auto endTime = std::chrono::steady_clock::now();
  result.latencyMs =
      std::chrono::duration<float, std::milli>(endTime - startTime).count();

  return result;
}

std::unique_ptr<LLMBackend> createOllamaBackend(llvm::StringRef endpoint) {
  return std::make_unique<OllamaBackend>(endpoint);
}

} // namespace mlir::iree_compiler::LLMAssist
