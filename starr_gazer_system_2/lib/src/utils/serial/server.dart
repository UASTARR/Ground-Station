import 'dart:async';
import 'dart:io';

class Server {
  late ServerSocket serverSocket;
  final List<Socket> clients = [];
  final StreamController<String> _controller = StreamController.broadcast();
  int? _port;

  Server({int? port}) {
    _port = port;
  }

  Future<void> dispose() async {
    await _controller.close();
    for (var client in clients) {
      await client.close();
    }
    await serverSocket.close();
  }

  Future<bool> startServer() async {
    try {
      serverSocket = await ServerSocket.bind(
        InternetAddress.loopbackIPv6,
        _port ?? 4040,
      );
      print("Assuming the role of server");
      print("Server listening on port ${serverSocket.port}");
      serverSocket.listen((Socket client) {
        clients.add(client);
        client.listen((data) {
          final message = String.fromCharCodes(data).trim();
          _controller.add(message); // Broadcast message to listeners
        }, onDone: () {
          clients.remove(client);
        });
      });
      return true;
    } on SocketException {
      return false;
    } catch (e) {
      throw Exception("Could not start server: $e");
    }
  }

  void broadcastMessage(String message, {Socket? exclude}) {
    for (var client in clients) {
      if (client != exclude) {
        client.write(message);
      }
    }
  }

  Stream<String> get stream => _controller.stream; // Expose the stream
}
