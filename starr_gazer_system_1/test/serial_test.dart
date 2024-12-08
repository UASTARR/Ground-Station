import 'package:flutter_test/flutter_test.dart';
import 'package:starr_gazer_system_1/src/utils/serial/serial.dart';

void main() {
  group('Serial Communication Integration Test', () {
    test('Server and Client can communicate', () async {
      // Initialize server and client
      final server = Serial(port: 4040);
      final client = Serial(port: 4040);

      // Lists to capture messages
      final serverMessages = <String>[];
      final clientMessages = <String>[];

      // Connect server and client
      await server.connect();
      expect(server.role, "server");
      await client.connect();
      expect(client.role, "client");

      // Listen for messages
      server.listen().listen((message) {
        serverMessages.add(message);
      });

      client.listen().listen((message) {
        clientMessages.add(message);
      });

      // Send messages
      server.sendMessage("Hello from server");
      client.sendMessage("Hello from client");

      // Wait for the communication to complete
      await Future.delayed(const Duration(seconds: 2));

      // Assertions
      expect(clientMessages, contains("Hello from server"));
      expect(serverMessages, contains("Hello from client"));

      // Clean up
      await server.dispose();
      await client.dispose();
    });
  });
}
