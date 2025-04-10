// screens/settings_screen.dart
import 'package:chronotopia_app/services/background_service.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  _SettingsScreenState createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  // Tracking settings
  bool _trackInLowBattery = true;
  int _locationAccuracy = 1; // 0: Low, 1: Balanced, 2: High
  int _uploadFrequency = 1; // 0: Low, 1: Balanced, 2: High
  bool _syncOnWifiOnly = false;
  bool _startOnBoot = true;
  bool _showNotification = true;
  bool _receiveServerMessages = true;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();

    setState(() {
      _trackInLowBattery = prefs.getBool('track_in_low_battery') ?? true;
      _locationAccuracy = prefs.getInt('location_accuracy') ?? 1;
      _uploadFrequency = prefs.getInt('upload_frequency') ?? 1;
      _syncOnWifiOnly = prefs.getBool('sync_on_wifi_only') ?? false;
      _startOnBoot = prefs.getBool('start_on_boot') ?? true;
      _showNotification = prefs.getBool('show_notification') ?? true;
      _receiveServerMessages = prefs.getBool('receive_server_messages') ?? true;
    });
  }

  Future<void> _saveSettings() async {
    final prefs = await SharedPreferences.getInstance();

    await prefs.setBool('track_in_low_battery', _trackInLowBattery);
    await prefs.setInt('location_accuracy', _locationAccuracy);
    await prefs.setInt('upload_frequency', _uploadFrequency);
    await prefs.setBool('sync_on_wifi_only', _syncOnWifiOnly);
    await prefs.setBool('start_on_boot', _startOnBoot);
    await prefs.setBool('show_notification', _showNotification);
    await prefs.setBool('receive_server_messages', _receiveServerMessages);

    // Apply settings to the service
    final backgroundService = Provider.of<BackgroundService>(
      context,
      listen: false,
    );
    await backgroundService.applySettings();

    // Show confirmation
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Settings saved')));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        children: [
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              'Tracking Settings',
              style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
            ),
          ),

          // Continue tracking in low battery
          SwitchListTile(
            title: const Text('Track in Low Battery Mode'),
            subtitle: const Text('Continue tracking when battery is below 20%'),
            value: _trackInLowBattery,
            onChanged: (value) {
              setState(() {
                _trackInLowBattery = value;
              });
            },
          ),

          // Location accuracy
          ListTile(
            title: const Text('Location Accuracy'),
            subtitle: Text(_getAccuracyText()),
            trailing: DropdownButton<int>(
              value: _locationAccuracy,
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _locationAccuracy = value;
                  });
                }
              },
              items: const [
                DropdownMenuItem(value: 0, child: Text('Low (Battery Saving)')),
                DropdownMenuItem(value: 1, child: Text('Balanced')),
                DropdownMenuItem(value: 2, child: Text('High (Best Accuracy)')),
              ],
            ),
          ),

          // Data upload frequency
          ListTile(
            title: const Text('Data Upload Frequency'),
            subtitle: Text(_getUploadFrequencyText()),
            trailing: DropdownButton<int>(
              value: _uploadFrequency,
              onChanged: (value) {
                if (value != null) {
                  setState(() {
                    _uploadFrequency = value;
                  });
                }
              },
              items: const [
                DropdownMenuItem(
                  value: 0,
                  child: Text('Low (Less Data Usage)'),
                ),
                DropdownMenuItem(value: 1, child: Text('Balanced')),
                DropdownMenuItem(value: 2, child: Text('High (Real-time)')),
              ],
            ),
          ),

          // Sync on Wi-Fi only
          SwitchListTile(
            title: const Text('Sync on Wi-Fi Only'),
            subtitle: const Text('Upload data only when connected to Wi-Fi'),
            value: _syncOnWifiOnly,
            onChanged: (value) {
              setState(() {
                _syncOnWifiOnly = value;
              });
            },
          ),

          const Divider(),

          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              'App Behavior',
              style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
            ),
          ),

          // Start on boot
          SwitchListTile(
            title: const Text('Start on Boot'),
            subtitle: const Text(
              'Automatically start tracking when device restarts',
            ),
            value: _startOnBoot,
            onChanged: (value) {
              setState(() {
                _startOnBoot = value;
              });
            },
          ),

          // Show notification
          SwitchListTile(
            title: const Text('Show Tracking Notification'),
            subtitle: const Text(
              'Display a persistent notification when tracking',
            ),
            value: _showNotification,
            onChanged: (value) {
              setState(() {
                _showNotification = value;
              });
            },
          ),

          // Receive server messages
          SwitchListTile(
            title: const Text('Receive Server Messages'),
            subtitle: const Text(
              'Allow server to send wake-up signals and updates',
            ),
            value: _receiveServerMessages,
            onChanged: (value) {
              setState(() {
                _receiveServerMessages = value;
              });
            },
          ),

          // Save button
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton(
              onPressed: _saveSettings,
              child: const Text('Save Settings'),
            ),
          ),

          // Data management section
          const Divider(),

          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              'Data Management',
              style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
            ),
          ),

          ListTile(
            title: const Text('Clear Local Cache'),
            subtitle: const Text('Delete temporary data stored on this device'),
            trailing: const Icon(Icons.delete_outline),
            onTap: _showClearCacheDialog,
          ),

          ListTile(
            title: const Text('Export Timeline Data'),
            subtitle: const Text('Save your timeline data as a file'),
            trailing: const Icon(Icons.file_download),
            onTap: () {
              // Implement export functionality
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Export functionality coming soon'),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  String _getAccuracyText() {
    switch (_locationAccuracy) {
      case 0:
        return 'Lower accuracy, better battery life';
      case 1:
        return 'Balanced accuracy and battery usage';
      case 2:
        return 'Highest accuracy, more battery usage';
      default:
        return '';
    }
  }

  String _getUploadFrequencyText() {
    switch (_uploadFrequency) {
      case 0:
        return 'Upload every few hours, less data usage';
      case 1:
        return 'Upload every 15-30 minutes';
      case 2:
        return 'Upload frequently, more data usage';
      default:
        return '';
    }
  }

  Future<void> _showClearCacheDialog() async {
    return showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Clear Local Cache'),
          content: const SingleChildScrollView(
            child: Text(
              'This will delete temporary data stored on your device. '
              'Uploaded data will still be available on your timeline. '
              'This operation cannot be undone.',
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Cancel'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
            TextButton(
              child: const Text('Clear Cache'),
              onPressed: () {
                // Implement cache clearing functionality
                Navigator.of(context).pop();
                ScaffoldMessenger.of(
                  context,
                ).showSnackBar(const SnackBar(content: Text('Cache cleared')));
              },
            ),
          ],
        );
      },
    );
  }
}
