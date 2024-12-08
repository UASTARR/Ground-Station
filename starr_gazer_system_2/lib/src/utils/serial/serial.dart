import 'dart:async';
import 'server.dart';
import 'client.dart';

class Serial {
  String role = "";
  int? _port;
  Server? _server;
  Client? _client;

  Serial({int? port}) {
    _port = port;
  }

  Future<void> connect() async {
    try {
      _server = Server(port: _port);
      bool? status = await _server?.startServer();

      if (status == false) {
        _server = null;
        _client = Client(port: _port);
        await _client!.startClient(); // Await the client's initialization
        role = "client";
      } else {
        role = "server";
        _client = null;
      }
    } catch (e) {
      throw Exception("Could not start server or client: $e");
    }
  }

  bool sendMessage(String message) {
    if (role == "server" && _server != null) {
      _server?.broadcastMessage(message);
      return true;
    } else if (role == "client" && _client != null) {
      _client?.sendMessage(message);
      return true;
    } else {
      return false;
    }
  }

  Stream<String> listen() {
    if (role == "server" && _server != null) {
      return _server!.stream; // Stream from Server
    } else if (role == "client" && _client != null) {
      return _client!.stream; // Stream from Client
    }
    throw Exception("No active server or client to listen to");
  }

  Future<void> dispose() async {
    await _server?.dispose();
    await _client?.dispose();
  }
}
