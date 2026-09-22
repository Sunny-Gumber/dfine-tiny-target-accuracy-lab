package com.sunnygumber.cctvaivisionlab.models

import com.sunnygumber.cctvaivisionlab.core.ModelDescriptor

object ModelCatalog {
    val yoloxSmall = ModelDescriptor(
        id = "yolox-small",
        fileName = "yolox_s.onnx",
        url = "https://huggingface.co/LibreYOLO/libreyolo-web/resolve/main/yolox_s.onnx?download=true",
        version = "libreyolo-web-main-256376a",
        sha256 = "b10957d5b66edd1d037c60cd47fdcfa8c60db67cae614039b1910f2878dfc081",
    )

    val relateAnything = ModelDescriptor(
        id = "relateanything-vits16plus",
        fileName = "relateanything.onnx",
        url = "https://huggingface.co/maelic/relsgg-vits16plus/resolve/main/relateanything.onnx?download=true",
        version = "relsgg-vits16plus-3edae20",
        sha256 = "b8b6a047c5e0771a897a5015c2ffb09d8fe5e3ffa0651e1e61af0e8436a3617a",
    )

    val predicateBank = ModelDescriptor(
        id = "relateanything-predicate-bank",
        fileName = "predicate_bank.npz",
        url = "https://huggingface.co/maelic/relsgg-vits16plus/resolve/main/predicate_bank.npz?download=true",
        version = "relsgg-vits16plus-87eab61",
        sha256 = "708f812d6579eab1be85b3378b79e376453c2d4f65b4fad9bcebb9f5bf05da06",
    )

    val required = listOf(yoloxSmall, relateAnything, predicateBank)
}
