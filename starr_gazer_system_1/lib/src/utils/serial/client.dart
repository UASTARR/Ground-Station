import 'dart:async';
import 'dart:io';

class Client {
  late Socket socket;
  final StreamController<String> _controller = StreamController.broadcast();
  int? _port;

  Client({int? port}) {
    _port = port;
  }

  Future<void> startClient() async {
    try {
      socket =
          await Socket.connect(InternetAddress.loopbackIPv6, _port ?? 4040);
      print("Assuming the role of client");
      print("Started Client on port ${socket.port}");
      socket.listen((data) {
        final message = String.fromCharCodes(data).trim();
        _controller.add(message); // Broadcast message to listeners
      }, onDone: () {
        socket.destroy();
      });
    } catch (e) {
      throw Exception("Could not start client: $e");
    }
  }

  Future<void> dispose() async {
    await _controller.close();
    await socket.close();
  }

  void sendMessage(String message) {
    if (message.isNotEmpty) {
      socket.write(message);
    } else {
      print("Socket is not connected or message is empty.");
    }
  }

  Stream<String> get stream => _controller.stream; // Expose the stream
}
