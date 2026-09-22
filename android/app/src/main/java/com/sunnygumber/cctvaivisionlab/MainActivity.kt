package com.sunnygumber.cctvaivisionlab

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.material3.MaterialTheme
import com.sunnygumber.cctvaivisionlab.ui.CctvAiApp
import com.sunnygumber.cctvaivisionlab.ui.VisionViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                val visionViewModel: VisionViewModel = viewModel()
                CctvAiApp(visionViewModel)
            }
        }
    }
}
