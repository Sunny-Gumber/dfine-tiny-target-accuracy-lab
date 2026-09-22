package com.sunnygumber.cctvaivisionlab.models

import com.sunnygumber.cctvaivisionlab.core.ModelDescriptor

object ModelCatalog {
    val yoloXSmall = ModelDescriptor(
        id = "yolox-s",
        fileName = "yolox_s.onnx",
        url = "https://huggingface.co/LibreYOLO/libreyolo-web/resolve/main/yolox_s.onnx",
        version = "libreyolo-web-main-yolox-s-v1",
    )

    val relateAnything = ModelDescriptor(
        id = "relateanything-vits16plus",
        fileName = "relateanything.onnx",
        url = "https://huggingface.co/maelic/relsgg-vits16plus/resolve/main/relateanything.onnx",
        version = "relsgg-vits16plus-main-v1",
    )

    val predicateBank = ModelDescriptor(
        id = "relateanything-predicate-bank",
        fileName = "predicate_bank.npz",
        url = "https://cdn.jsdelivr.net/gh/Maelic/RelateAnything@main/deploy/dist/relsgg-vits16plus/predicate_bank.npz",
        version = "relsgg-vits16plus-bank-main-v1",
    )

    val all = listOf(yoloXSmall, relateAnything, predicateBank)
}
