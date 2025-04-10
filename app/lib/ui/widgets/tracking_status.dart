// widgets/tracking_status_card.dart
import 'package:flutter/material.dart';

class TrackingStatusCard extends StatelessWidget {
  final bool isTracking;
  final Function(bool) onToggle;

  const TrackingStatusCard({
    super.key,
    required this.isTracking,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: isTracking ? Colors.green.shade50 : Colors.red.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Location Tracking',
                  style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
                ),
                Switch(
                  value: isTracking,
                  onChanged: onToggle,
                  activeColor: Colors.green,
                ),
              ],
            ),
            const SizedBox(height: 8.0),
            Text(
              isTracking
                  ? 'Tracking is active. Your timeline is being built.'
                  : 'Tracking is paused. No data is being collected.',
              style: TextStyle(
                color: isTracking ? Colors.green.shade700 : Colors.red.shade700,
              ),
            ),
            const SizedBox(height: 16.0),
            Text(
              isTracking
                  ? 'Last update: ${DateTime.now().toString().substring(0, 16)}'
                  : 'Enable tracking to start building your timeline',
              style: const TextStyle(
                fontSize: 12.0,
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
