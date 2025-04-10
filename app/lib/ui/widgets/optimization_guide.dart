// widgets/optimization_guide.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'package:url_launcher/url_launcher.dart';

class OptimizationGuide extends StatelessWidget {
  final String manufacturer;

  const OptimizationGuide({super.key, required this.manufacturer});

  @override
  Widget build(BuildContext context) {
    if (Platform.isAndroid) {
      return _buildAndroidGuide(context);
    } else if (Platform.isIOS) {
      return _buildIOSGuide(context);
    } else {
      return const SizedBox.shrink();
    }
  }

  Widget _buildAndroidGuide(BuildContext context) {
    // Determine manufacturer-specific instructions
    List<Map<String, dynamic>> steps = _getManufacturerSteps();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Optimize for Best Tracking',
              style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8.0),
            const Text(
              'To ensure reliable background tracking, please adjust these device settings:',
            ),
            const SizedBox(height: 16.0),
            ...steps.map((step) => _buildStepItem(context, step)),
            const SizedBox(height: 8.0),
            const Divider(),
            TextButton.icon(
              onPressed: () {
                _openBatteryOptimizationSettings();
              },
              icon: const Icon(Icons.power_settings_new),
              label: const Text('Open Battery Settings'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildIOSGuide(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Optimize for Best Tracking',
              style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8.0),
            const Text(
              'To ensure reliable background tracking, please adjust these iOS settings:',
            ),
            const SizedBox(height: 16.0),
            _buildStepItem(context, {
              'title': 'Enable "Always" Location Permission',
              'description':
                  'Go to Settings → Privacy → Location Services → Trip Tracker → Select "Always"',
              'icon': Icons.location_on,
            }),
            _buildStepItem(context, {
              'title': 'Enable Background App Refresh',
              'description':
                  'Go to Settings → General → Background App Refresh → Enable for Trip Tracker',
              'icon': Icons.refresh,
            }),
            const SizedBox(height: 8.0),
            const Divider(),
            TextButton.icon(
              onPressed: () {
                _openLocationSettings();
              },
              icon: const Icon(Icons.settings),
              label: const Text('Open Location Settings'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStepItem(BuildContext context, Map<String, dynamic> step) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(step['icon'] ?? Icons.check_circle_outline, color: Colors.blue),
          const SizedBox(width: 16.0),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  step['title'],
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4.0),
                Text(
                  step['description'],
                  style: const TextStyle(fontSize: 13.0),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  List<Map<String, dynamic>> _getManufacturerSteps() {
    // Default steps for all Android devices
    List<Map<String, dynamic>> steps = [
      {
        'title': 'Disable Battery Optimization',
        'description':
            'Go to Settings → Apps → Trip Tracker → Battery → Unrestricted',
        'icon': Icons.battery_charging_full,
      },
      {
        'title': 'Enable Autostart Permission',
        'description':
            'Allow Trip Tracker to start automatically after device reboot',
        'icon': Icons.play_circle_outline,
      },
    ];

    // Add manufacturer-specific steps
    if (manufacturer.contains('xiaomi') || manufacturer.contains('redmi')) {
      steps.add({
        'title': 'Xiaomi/MIUI Settings',
        'description':
            'Open Security app → Permissions → Autostart → Enable for Trip Tracker. Also check "Battery saver" settings.',
        'icon': Icons.security,
      });
    } else if (manufacturer.contains('samsung')) {
      steps.add({
        'title': 'Samsung Settings',
        'description':
            'Go to Settings → Device Care → Battery → Background usage limits → Add Trip Tracker to "Unmonitored apps"',
        'icon': Icons.phonelink_setup,
      });
    } else if (manufacturer.contains('huawei')) {
      steps.add({
        'title': 'Huawei Settings',
        'description':
            'Go to Settings → Battery → App launch → Trip Tracker → Manage manually and disable all restrictions',
        'icon': Icons.phonelink_setup,
      });
    } else if (manufacturer.contains('oppo') ||
        manufacturer.contains('oneplus') ||
        manufacturer.contains('realme')) {
      steps.add({
        'title': 'OxygenOS/ColorOS Settings',
        'description':
            'Go to Settings → Battery → Battery Optimization → Find Trip Tracker → Don\'t optimize',
        'icon': Icons.phonelink_setup,
      });
    }

    return steps;
  }

  void _openBatteryOptimizationSettings() async {
    try {
      if (Platform.isAndroid) {
        const platform = MethodChannel('io.chronotopia.app/settings');
        await platform.invokeMethod('openBatteryOptimizationSettings');
      }
    } catch (e) {
      // Fallback for when the specific method isn't available
      if (manufacturer.contains('xiaomi')) {
        _launchUri('miui://security');
      } else {
        // Generic intent
        await _launchUri('package:io.chronotopia.app');
      }
    }
  }

  void _openLocationSettings() async {
    if (Platform.isIOS) {
      const uri = 'app-settings:';
      await _launchUri(uri);
    }
  }

  Future<void> _launchUri(String uris) async {
    final uri = Uri.parse(uris);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    }
  }
}
