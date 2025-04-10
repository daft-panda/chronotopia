import 'package:chronotopia_app/services/background_service.dart';
import 'package:chronotopia_app/ui/screens/settings.dart';
import 'package:chronotopia_app/ui/screens/timeline.dart';
import 'package:chronotopia_app/ui/widgets/optimization_guide.dart';
import 'package:chronotopia_app/ui/widgets/tracking_status.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'dart:io';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _deviceModel = 'Unknown device';
  String _osVersion = 'Unknown OS';
  String _manufacturer = 'Unknown manufacturer';

  @override
  void initState() {
    super.initState();
    _getDeviceInfo();
  }

  Future<void> _getDeviceInfo() async {
    final deviceInfo = DeviceInfoPlugin();

    if (Platform.isAndroid) {
      final androidInfo = await deviceInfo.androidInfo;
      setState(() {
        _deviceModel = androidInfo.model;
        _osVersion = 'Android ${androidInfo.version.release}';
        _manufacturer = androidInfo.manufacturer;
      });
    } else if (Platform.isIOS) {
      final iosInfo = await deviceInfo.iosInfo;
      setState(() {
        _deviceModel = iosInfo.model;
        _osVersion = '${iosInfo.systemName} ${iosInfo.systemVersion}';
        _manufacturer = 'Apple';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final backgroundService = Provider.of<BackgroundService>(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Trip Tracker'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const SettingsScreen()),
              );
            },
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async {
          await _getDeviceInfo();
        },
        child: ListView(
          padding: const EdgeInsets.all(16.0),
          children: [
            // Tracking status card
            TrackingStatusCard(
              isTracking: backgroundService.isRunning,
              onToggle: (enabled) {
                if (enabled) {
                  backgroundService.startTracking();
                } else {
                  backgroundService.stopTracking();
                }
              },
            ),

            const SizedBox(height: 16.0),

            // Device info card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Device Information',
                      style: TextStyle(
                        fontSize: 18.0,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8.0),
                    Text('Model: $_deviceModel'),
                    Text('OS: $_osVersion'),
                    Text('Manufacturer: $_manufacturer'),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16.0),

            // Battery Optimization Guide
            OptimizationGuide(manufacturer: _manufacturer.toLowerCase()),

            const SizedBox(height: 16.0),

            // View Timeline Button
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const TimelineScreen(),
                  ),
                );
              },
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 12.0),
              ),
              child: const Text('View Your Timeline'),
            ),

            const SizedBox(height: 24.0),

            // Information about background tracking
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'About Background Tracking',
                      style: TextStyle(
                        fontSize: 18.0,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8.0),
                    Text(
                      'Trip Tracker needs to run in the background to create your timeline. '
                      'Please follow the optimization guide above to ensure reliable tracking. '
                      'The app uses your location and activity data to build a timeline of your trips and visited places.',
                    ),
                    SizedBox(height: 8.0),
                    Text(
                      'Your data is stored securely and only used to provide timeline features.',
                      style: TextStyle(fontStyle: FontStyle.italic),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
