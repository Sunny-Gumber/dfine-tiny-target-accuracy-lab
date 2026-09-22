package com.sunnygumber.cctvaivisionlab.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CctvAiApp() {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("CCTV AI Vision Lab") },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text(
                text = "Native Android V1",
                style = MaterialTheme.typography.headlineSmall,
            )
            Text(
                text = "Architecture scaffold is active. CameraX and native ONNX inference land in the next workstreams.",
                style = MaterialTheme.typography.bodyLarge,
            )
            HorizontalDivider()
            StatusLine("Detector", "interface ready")
            StatusLine("Relation engine", "interface ready")
            StatusLine("Model manager", "interface ready")
            StatusLine("Tracker", "interface ready")
            StatusLine("Frame scheduler", "interface ready")
        }
    }
}

@Composable
private fun StatusLine(label: String, value: String) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(label, style = MaterialTheme.typography.labelLarge)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}
