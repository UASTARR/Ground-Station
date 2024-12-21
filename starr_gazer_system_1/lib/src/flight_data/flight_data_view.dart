import 'package:flutter/material.dart';
import 'package:starr_gazer_system_1/src/custom_app_bar/custom_app_bar.dart';
import '../settings/settings_view.dart';
import 'package:flutter_gen/gen_l10n/app_localizations.dart';

class FlightDataView extends StatelessWidget {
  const FlightDataView({
    super.key,
  });

  static const routeName = '/';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(
        title: AppLocalizations.of(context)!.appTitle,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.restorablePushNamed(context, SettingsView.routeName);
            },
          ),
        ],
      ),

      body: const Text("Temp")
    );
  }
}
