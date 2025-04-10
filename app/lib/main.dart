import 'package:chronotopia_app/services/background_service.dart';
import 'package:chronotopia_app/ui/screens/home.dart';
import 'package:chronotopia_app/utils/permission_handler.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize background service early
  final backgroundService = BackgroundService();
  await backgroundService.initialize();

  runApp(
    MultiProvider(
      providers: [ChangeNotifierProvider(create: (_) => backgroundService)],
      child: const TripTrackerApp(),
    ),
  );
}

class TripTrackerApp extends StatefulWidget {
  const TripTrackerApp({super.key});

  @override
  _TripTrackerAppState createState() => _TripTrackerAppState();
}

class _TripTrackerAppState extends State<TripTrackerApp> {
  @override
  void initState() {
    super.initState();
    _checkAndRequestPermissions();
  }

  Future<void> _checkAndRequestPermissions() async {
    final permissionHandler = PermissionHandler();
    await permissionHandler.requestRequiredPermissions();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Chronotopia',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomeScreen(),
    );
  }
}
