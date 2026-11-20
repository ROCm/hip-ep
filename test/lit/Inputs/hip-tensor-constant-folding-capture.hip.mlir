module attributes {hipdnn.input_count = 1 : i64, hipdnn.input_element_sizes = array<i64: 4>, hipdnn.input_shapes = [array<i64: 1, 54400>], hipdnn.output_count = 1 : i64, hipdnn.output_element_sizes = array<i64: 4>, hipdnn.output_shapes = [array<i64: 1, 169, 32>], torch_hip.importer = "export", torch_hip.torch_mlir_commit = "69a686fa3c1c68bc4b1bfedb754eaebb4ce22407", torch_hip.weights_file = "facebook-data2vec-audio-base-960h.weights.bin", torch_hip.weights_index = "facebook-data2vec-audio-base-960h.weights.json"} {
  func.func @main_graph(%arg0: !hip.context, %arg1: tensor<1x54400xf32>) -> tensor<1x169x32xf32> {
    %0 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 372752384 : i64, serialization_order = 0 : i64, size = 128 : i64, source_name = "torch_hip_weight_372752384_128"} : tensor<32xf32>
    %1 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 372654080 : i64, serialization_order = 1 : i64, size = 98304 : i64, source_name = "torch_hip_weight_372654080_98304"} : tensor<32x768xf32>
    %2 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 372651008 : i64, serialization_order = 2 : i64, size = 3072 : i64, source_name = "torch_hip_weight_372651008_3072"} : tensor<768xf32>
    %3 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 372647936 : i64, serialization_order = 3 : i64, size = 3072 : i64, source_name = "torch_hip_weight_372647936_3072"} : tensor<768xf32>
    %4 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 372644864 : i64, serialization_order = 4 : i64, size = 3072 : i64, source_name = "torch_hip_weight_372644864_3072"} : tensor<768xf32>
    %5 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 363207680 : i64, serialization_order = 5 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_363207680_9437184"} : tensor<768x3072xf32>
    %6 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 363195392 : i64, serialization_order = 6 : i64, size = 12288 : i64, source_name = "torch_hip_weight_363195392_12288"} : tensor<3072xf32>
    %7 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 353758208 : i64, serialization_order = 7 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_353758208_9437184"} : tensor<3072x768xf32>
    %8 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 353755136 : i64, serialization_order = 8 : i64, size = 3072 : i64, source_name = "torch_hip_weight_353755136_3072"} : tensor<768xf32>
    %9 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 353752064 : i64, serialization_order = 9 : i64, size = 3072 : i64, source_name = "torch_hip_weight_353752064_3072"} : tensor<768xf32>
    %10 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 353748992 : i64, serialization_order = 10 : i64, size = 3072 : i64, source_name = "torch_hip_weight_353748992_3072"} : tensor<768xf32>
    %11 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 351389696 : i64, serialization_order = 11 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_351389696_2359296"} : tensor<768x768xf32>
    %12 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 351386624 : i64, serialization_order = 12 : i64, size = 3072 : i64, source_name = "torch_hip_weight_351386624_3072"} : tensor<768xf32>
    %13 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 349027328 : i64, serialization_order = 13 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_349027328_2359296"} : tensor<768x768xf32>
    %14 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 349024256 : i64, serialization_order = 14 : i64, size = 3072 : i64, source_name = "torch_hip_weight_349024256_3072"} : tensor<768xf32>
    %15 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 346664960 : i64, serialization_order = 15 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_346664960_2359296"} : tensor<768x768xf32>
    %16 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 346661888 : i64, serialization_order = 16 : i64, size = 3072 : i64, source_name = "torch_hip_weight_346661888_3072"} : tensor<768xf32>
    %17 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 344302592 : i64, serialization_order = 17 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_344302592_2359296"} : tensor<768x768xf32>
    %18 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 344299520 : i64, serialization_order = 18 : i64, size = 3072 : i64, source_name = "torch_hip_weight_344299520_3072"} : tensor<768xf32>
    %19 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 344296448 : i64, serialization_order = 19 : i64, size = 3072 : i64, source_name = "torch_hip_weight_344296448_3072"} : tensor<768xf32>
    %20 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 344293376 : i64, serialization_order = 20 : i64, size = 3072 : i64, source_name = "torch_hip_weight_344293376_3072"} : tensor<768xf32>
    %21 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 334856192 : i64, serialization_order = 21 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_334856192_9437184"} : tensor<768x3072xf32>
    %22 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 334843904 : i64, serialization_order = 22 : i64, size = 12288 : i64, source_name = "torch_hip_weight_334843904_12288"} : tensor<3072xf32>
    %23 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 325406720 : i64, serialization_order = 23 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_325406720_9437184"} : tensor<3072x768xf32>
    %24 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 325403648 : i64, serialization_order = 24 : i64, size = 3072 : i64, source_name = "torch_hip_weight_325403648_3072"} : tensor<768xf32>
    %25 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 325400576 : i64, serialization_order = 25 : i64, size = 3072 : i64, source_name = "torch_hip_weight_325400576_3072"} : tensor<768xf32>
    %26 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 325397504 : i64, serialization_order = 26 : i64, size = 3072 : i64, source_name = "torch_hip_weight_325397504_3072"} : tensor<768xf32>
    %27 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 323038208 : i64, serialization_order = 27 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_323038208_2359296"} : tensor<768x768xf32>
    %28 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 323035136 : i64, serialization_order = 28 : i64, size = 3072 : i64, source_name = "torch_hip_weight_323035136_3072"} : tensor<768xf32>
    %29 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 320675840 : i64, serialization_order = 29 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_320675840_2359296"} : tensor<768x768xf32>
    %30 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 320672768 : i64, serialization_order = 30 : i64, size = 3072 : i64, source_name = "torch_hip_weight_320672768_3072"} : tensor<768xf32>
    %31 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 318313472 : i64, serialization_order = 31 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_318313472_2359296"} : tensor<768x768xf32>
    %32 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 318310400 : i64, serialization_order = 32 : i64, size = 3072 : i64, source_name = "torch_hip_weight_318310400_3072"} : tensor<768xf32>
    %33 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 315951104 : i64, serialization_order = 33 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_315951104_2359296"} : tensor<768x768xf32>
    %34 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 315948032 : i64, serialization_order = 34 : i64, size = 3072 : i64, source_name = "torch_hip_weight_315948032_3072"} : tensor<768xf32>
    %35 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 315944960 : i64, serialization_order = 35 : i64, size = 3072 : i64, source_name = "torch_hip_weight_315944960_3072"} : tensor<768xf32>
    %36 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 315941888 : i64, serialization_order = 36 : i64, size = 3072 : i64, source_name = "torch_hip_weight_315941888_3072"} : tensor<768xf32>
    %37 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 306504704 : i64, serialization_order = 37 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_306504704_9437184"} : tensor<768x3072xf32>
    %38 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 306492416 : i64, serialization_order = 38 : i64, size = 12288 : i64, source_name = "torch_hip_weight_306492416_12288"} : tensor<3072xf32>
    %39 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 297055232 : i64, serialization_order = 39 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_297055232_9437184"} : tensor<3072x768xf32>
    %40 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 297052160 : i64, serialization_order = 40 : i64, size = 3072 : i64, source_name = "torch_hip_weight_297052160_3072"} : tensor<768xf32>
    %41 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 297049088 : i64, serialization_order = 41 : i64, size = 3072 : i64, source_name = "torch_hip_weight_297049088_3072"} : tensor<768xf32>
    %42 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 297046016 : i64, serialization_order = 42 : i64, size = 3072 : i64, source_name = "torch_hip_weight_297046016_3072"} : tensor<768xf32>
    %43 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 294686720 : i64, serialization_order = 43 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_294686720_2359296"} : tensor<768x768xf32>
    %44 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 294683648 : i64, serialization_order = 44 : i64, size = 3072 : i64, source_name = "torch_hip_weight_294683648_3072"} : tensor<768xf32>
    %45 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 292324352 : i64, serialization_order = 45 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_292324352_2359296"} : tensor<768x768xf32>
    %46 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 292321280 : i64, serialization_order = 46 : i64, size = 3072 : i64, source_name = "torch_hip_weight_292321280_3072"} : tensor<768xf32>
    %47 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 289961984 : i64, serialization_order = 47 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_289961984_2359296"} : tensor<768x768xf32>
    %48 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 289958912 : i64, serialization_order = 48 : i64, size = 3072 : i64, source_name = "torch_hip_weight_289958912_3072"} : tensor<768xf32>
    %49 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 287599616 : i64, serialization_order = 49 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_287599616_2359296"} : tensor<768x768xf32>
    %50 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 287596544 : i64, serialization_order = 50 : i64, size = 3072 : i64, source_name = "torch_hip_weight_287596544_3072"} : tensor<768xf32>
    %51 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 287593472 : i64, serialization_order = 51 : i64, size = 3072 : i64, source_name = "torch_hip_weight_287593472_3072"} : tensor<768xf32>
    %52 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 287590400 : i64, serialization_order = 52 : i64, size = 3072 : i64, source_name = "torch_hip_weight_287590400_3072"} : tensor<768xf32>
    %53 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 278153216 : i64, serialization_order = 53 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_278153216_9437184"} : tensor<768x3072xf32>
    %54 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 278140928 : i64, serialization_order = 54 : i64, size = 12288 : i64, source_name = "torch_hip_weight_278140928_12288"} : tensor<3072xf32>
    %55 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 268703744 : i64, serialization_order = 55 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_268703744_9437184"} : tensor<3072x768xf32>
    %56 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 268700672 : i64, serialization_order = 56 : i64, size = 3072 : i64, source_name = "torch_hip_weight_268700672_3072"} : tensor<768xf32>
    %57 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 268697600 : i64, serialization_order = 57 : i64, size = 3072 : i64, source_name = "torch_hip_weight_268697600_3072"} : tensor<768xf32>
    %58 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 268694528 : i64, serialization_order = 58 : i64, size = 3072 : i64, source_name = "torch_hip_weight_268694528_3072"} : tensor<768xf32>
    %59 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 266335232 : i64, serialization_order = 59 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_266335232_2359296"} : tensor<768x768xf32>
    %60 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 266332160 : i64, serialization_order = 60 : i64, size = 3072 : i64, source_name = "torch_hip_weight_266332160_3072"} : tensor<768xf32>
    %61 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 263972864 : i64, serialization_order = 61 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_263972864_2359296"} : tensor<768x768xf32>
    %62 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 263969792 : i64, serialization_order = 62 : i64, size = 3072 : i64, source_name = "torch_hip_weight_263969792_3072"} : tensor<768xf32>
    %63 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 261610496 : i64, serialization_order = 63 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_261610496_2359296"} : tensor<768x768xf32>
    %64 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 261607424 : i64, serialization_order = 64 : i64, size = 3072 : i64, source_name = "torch_hip_weight_261607424_3072"} : tensor<768xf32>
    %65 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 259248128 : i64, serialization_order = 65 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_259248128_2359296"} : tensor<768x768xf32>
    %66 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 259245056 : i64, serialization_order = 66 : i64, size = 3072 : i64, source_name = "torch_hip_weight_259245056_3072"} : tensor<768xf32>
    %67 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 259241984 : i64, serialization_order = 67 : i64, size = 3072 : i64, source_name = "torch_hip_weight_259241984_3072"} : tensor<768xf32>
    %68 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 259238912 : i64, serialization_order = 68 : i64, size = 3072 : i64, source_name = "torch_hip_weight_259238912_3072"} : tensor<768xf32>
    %69 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 249801728 : i64, serialization_order = 69 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_249801728_9437184"} : tensor<768x3072xf32>
    %70 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 249789440 : i64, serialization_order = 70 : i64, size = 12288 : i64, source_name = "torch_hip_weight_249789440_12288"} : tensor<3072xf32>
    %71 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 240352256 : i64, serialization_order = 71 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_240352256_9437184"} : tensor<3072x768xf32>
    %72 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 240349184 : i64, serialization_order = 72 : i64, size = 3072 : i64, source_name = "torch_hip_weight_240349184_3072"} : tensor<768xf32>
    %73 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 240346112 : i64, serialization_order = 73 : i64, size = 3072 : i64, source_name = "torch_hip_weight_240346112_3072"} : tensor<768xf32>
    %74 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 240343040 : i64, serialization_order = 74 : i64, size = 3072 : i64, source_name = "torch_hip_weight_240343040_3072"} : tensor<768xf32>
    %75 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 237983744 : i64, serialization_order = 75 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_237983744_2359296"} : tensor<768x768xf32>
    %76 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 237980672 : i64, serialization_order = 76 : i64, size = 3072 : i64, source_name = "torch_hip_weight_237980672_3072"} : tensor<768xf32>
    %77 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 235621376 : i64, serialization_order = 77 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_235621376_2359296"} : tensor<768x768xf32>
    %78 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 235618304 : i64, serialization_order = 78 : i64, size = 3072 : i64, source_name = "torch_hip_weight_235618304_3072"} : tensor<768xf32>
    %79 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 233259008 : i64, serialization_order = 79 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_233259008_2359296"} : tensor<768x768xf32>
    %80 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 233255936 : i64, serialization_order = 80 : i64, size = 3072 : i64, source_name = "torch_hip_weight_233255936_3072"} : tensor<768xf32>
    %81 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 230896640 : i64, serialization_order = 81 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_230896640_2359296"} : tensor<768x768xf32>
    %82 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 230893568 : i64, serialization_order = 82 : i64, size = 3072 : i64, source_name = "torch_hip_weight_230893568_3072"} : tensor<768xf32>
    %83 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 230890496 : i64, serialization_order = 83 : i64, size = 3072 : i64, source_name = "torch_hip_weight_230890496_3072"} : tensor<768xf32>
    %84 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 230887424 : i64, serialization_order = 84 : i64, size = 3072 : i64, source_name = "torch_hip_weight_230887424_3072"} : tensor<768xf32>
    %85 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 221450240 : i64, serialization_order = 85 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_221450240_9437184"} : tensor<768x3072xf32>
    %86 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 221437952 : i64, serialization_order = 86 : i64, size = 12288 : i64, source_name = "torch_hip_weight_221437952_12288"} : tensor<3072xf32>
    %87 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 212000768 : i64, serialization_order = 87 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_212000768_9437184"} : tensor<3072x768xf32>
    %88 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 211997696 : i64, serialization_order = 88 : i64, size = 3072 : i64, source_name = "torch_hip_weight_211997696_3072"} : tensor<768xf32>
    %89 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 211994624 : i64, serialization_order = 89 : i64, size = 3072 : i64, source_name = "torch_hip_weight_211994624_3072"} : tensor<768xf32>
    %90 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 211991552 : i64, serialization_order = 90 : i64, size = 3072 : i64, source_name = "torch_hip_weight_211991552_3072"} : tensor<768xf32>
    %91 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 209632256 : i64, serialization_order = 91 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_209632256_2359296"} : tensor<768x768xf32>
    %92 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 209629184 : i64, serialization_order = 92 : i64, size = 3072 : i64, source_name = "torch_hip_weight_209629184_3072"} : tensor<768xf32>
    %93 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 207269888 : i64, serialization_order = 93 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_207269888_2359296"} : tensor<768x768xf32>
    %94 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 207266816 : i64, serialization_order = 94 : i64, size = 3072 : i64, source_name = "torch_hip_weight_207266816_3072"} : tensor<768xf32>
    %95 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 204907520 : i64, serialization_order = 95 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_204907520_2359296"} : tensor<768x768xf32>
    %96 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 204904448 : i64, serialization_order = 96 : i64, size = 3072 : i64, source_name = "torch_hip_weight_204904448_3072"} : tensor<768xf32>
    %97 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 202545152 : i64, serialization_order = 97 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_202545152_2359296"} : tensor<768x768xf32>
    %98 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 202542080 : i64, serialization_order = 98 : i64, size = 3072 : i64, source_name = "torch_hip_weight_202542080_3072"} : tensor<768xf32>
    %99 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 202539008 : i64, serialization_order = 99 : i64, size = 3072 : i64, source_name = "torch_hip_weight_202539008_3072"} : tensor<768xf32>
    %100 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 202535936 : i64, serialization_order = 100 : i64, size = 3072 : i64, source_name = "torch_hip_weight_202535936_3072"} : tensor<768xf32>
    %101 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 193098752 : i64, serialization_order = 101 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_193098752_9437184"} : tensor<768x3072xf32>
    %102 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 193086464 : i64, serialization_order = 102 : i64, size = 12288 : i64, source_name = "torch_hip_weight_193086464_12288"} : tensor<3072xf32>
    %103 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 183649280 : i64, serialization_order = 103 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_183649280_9437184"} : tensor<3072x768xf32>
    %104 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 183646208 : i64, serialization_order = 104 : i64, size = 3072 : i64, source_name = "torch_hip_weight_183646208_3072"} : tensor<768xf32>
    %105 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 183643136 : i64, serialization_order = 105 : i64, size = 3072 : i64, source_name = "torch_hip_weight_183643136_3072"} : tensor<768xf32>
    %106 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 183640064 : i64, serialization_order = 106 : i64, size = 3072 : i64, source_name = "torch_hip_weight_183640064_3072"} : tensor<768xf32>
    %107 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 181280768 : i64, serialization_order = 107 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_181280768_2359296"} : tensor<768x768xf32>
    %108 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 181277696 : i64, serialization_order = 108 : i64, size = 3072 : i64, source_name = "torch_hip_weight_181277696_3072"} : tensor<768xf32>
    %109 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 178918400 : i64, serialization_order = 109 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_178918400_2359296"} : tensor<768x768xf32>
    %110 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 178915328 : i64, serialization_order = 110 : i64, size = 3072 : i64, source_name = "torch_hip_weight_178915328_3072"} : tensor<768xf32>
    %111 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 176556032 : i64, serialization_order = 111 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_176556032_2359296"} : tensor<768x768xf32>
    %112 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 176552960 : i64, serialization_order = 112 : i64, size = 3072 : i64, source_name = "torch_hip_weight_176552960_3072"} : tensor<768xf32>
    %113 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 174193664 : i64, serialization_order = 113 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_174193664_2359296"} : tensor<768x768xf32>
    %114 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 174190592 : i64, serialization_order = 114 : i64, size = 3072 : i64, source_name = "torch_hip_weight_174190592_3072"} : tensor<768xf32>
    %115 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 174187520 : i64, serialization_order = 115 : i64, size = 3072 : i64, source_name = "torch_hip_weight_174187520_3072"} : tensor<768xf32>
    %116 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 174184448 : i64, serialization_order = 116 : i64, size = 3072 : i64, source_name = "torch_hip_weight_174184448_3072"} : tensor<768xf32>
    %117 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 164747264 : i64, serialization_order = 117 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_164747264_9437184"} : tensor<768x3072xf32>
    %118 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 164734976 : i64, serialization_order = 118 : i64, size = 12288 : i64, source_name = "torch_hip_weight_164734976_12288"} : tensor<3072xf32>
    %119 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 155297792 : i64, serialization_order = 119 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_155297792_9437184"} : tensor<3072x768xf32>
    %120 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 155294720 : i64, serialization_order = 120 : i64, size = 3072 : i64, source_name = "torch_hip_weight_155294720_3072"} : tensor<768xf32>
    %121 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 155291648 : i64, serialization_order = 121 : i64, size = 3072 : i64, source_name = "torch_hip_weight_155291648_3072"} : tensor<768xf32>
    %122 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 155288576 : i64, serialization_order = 122 : i64, size = 3072 : i64, source_name = "torch_hip_weight_155288576_3072"} : tensor<768xf32>
    %123 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 152929280 : i64, serialization_order = 123 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_152929280_2359296"} : tensor<768x768xf32>
    %124 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 152926208 : i64, serialization_order = 124 : i64, size = 3072 : i64, source_name = "torch_hip_weight_152926208_3072"} : tensor<768xf32>
    %125 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 150566912 : i64, serialization_order = 125 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_150566912_2359296"} : tensor<768x768xf32>
    %126 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 150563840 : i64, serialization_order = 126 : i64, size = 3072 : i64, source_name = "torch_hip_weight_150563840_3072"} : tensor<768xf32>
    %127 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 148204544 : i64, serialization_order = 127 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_148204544_2359296"} : tensor<768x768xf32>
    %128 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 148201472 : i64, serialization_order = 128 : i64, size = 3072 : i64, source_name = "torch_hip_weight_148201472_3072"} : tensor<768xf32>
    %129 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 145842176 : i64, serialization_order = 129 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_145842176_2359296"} : tensor<768x768xf32>
    %130 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 145839104 : i64, serialization_order = 130 : i64, size = 3072 : i64, source_name = "torch_hip_weight_145839104_3072"} : tensor<768xf32>
    %131 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 145836032 : i64, serialization_order = 131 : i64, size = 3072 : i64, source_name = "torch_hip_weight_145836032_3072"} : tensor<768xf32>
    %132 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 145832960 : i64, serialization_order = 132 : i64, size = 3072 : i64, source_name = "torch_hip_weight_145832960_3072"} : tensor<768xf32>
    %133 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 136395776 : i64, serialization_order = 133 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_136395776_9437184"} : tensor<768x3072xf32>
    %134 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 136383488 : i64, serialization_order = 134 : i64, size = 12288 : i64, source_name = "torch_hip_weight_136383488_12288"} : tensor<3072xf32>
    %135 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 126946304 : i64, serialization_order = 135 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_126946304_9437184"} : tensor<3072x768xf32>
    %136 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 126943232 : i64, serialization_order = 136 : i64, size = 3072 : i64, source_name = "torch_hip_weight_126943232_3072"} : tensor<768xf32>
    %137 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 126940160 : i64, serialization_order = 137 : i64, size = 3072 : i64, source_name = "torch_hip_weight_126940160_3072"} : tensor<768xf32>
    %138 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 126937088 : i64, serialization_order = 138 : i64, size = 3072 : i64, source_name = "torch_hip_weight_126937088_3072"} : tensor<768xf32>
    %139 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 124577792 : i64, serialization_order = 139 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_124577792_2359296"} : tensor<768x768xf32>
    %140 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 124574720 : i64, serialization_order = 140 : i64, size = 3072 : i64, source_name = "torch_hip_weight_124574720_3072"} : tensor<768xf32>
    %141 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 122215424 : i64, serialization_order = 141 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_122215424_2359296"} : tensor<768x768xf32>
    %142 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 122212352 : i64, serialization_order = 142 : i64, size = 3072 : i64, source_name = "torch_hip_weight_122212352_3072"} : tensor<768xf32>
    %143 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 119853056 : i64, serialization_order = 143 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_119853056_2359296"} : tensor<768x768xf32>
    %144 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 119849984 : i64, serialization_order = 144 : i64, size = 3072 : i64, source_name = "torch_hip_weight_119849984_3072"} : tensor<768xf32>
    %145 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 117490688 : i64, serialization_order = 145 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_117490688_2359296"} : tensor<768x768xf32>
    %146 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 117487616 : i64, serialization_order = 146 : i64, size = 3072 : i64, source_name = "torch_hip_weight_117487616_3072"} : tensor<768xf32>
    %147 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 117484544 : i64, serialization_order = 147 : i64, size = 3072 : i64, source_name = "torch_hip_weight_117484544_3072"} : tensor<768xf32>
    %148 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 117481472 : i64, serialization_order = 148 : i64, size = 3072 : i64, source_name = "torch_hip_weight_117481472_3072"} : tensor<768xf32>
    %149 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 108044288 : i64, serialization_order = 149 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_108044288_9437184"} : tensor<768x3072xf32>
    %150 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 108032000 : i64, serialization_order = 150 : i64, size = 12288 : i64, source_name = "torch_hip_weight_108032000_12288"} : tensor<3072xf32>
    %151 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 98594816 : i64, serialization_order = 151 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_98594816_9437184"} : tensor<3072x768xf32>
    %152 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 98591744 : i64, serialization_order = 152 : i64, size = 3072 : i64, source_name = "torch_hip_weight_98591744_3072"} : tensor<768xf32>
    %153 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 98588672 : i64, serialization_order = 153 : i64, size = 3072 : i64, source_name = "torch_hip_weight_98588672_3072"} : tensor<768xf32>
    %154 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 98585600 : i64, serialization_order = 154 : i64, size = 3072 : i64, source_name = "torch_hip_weight_98585600_3072"} : tensor<768xf32>
    %155 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 96226304 : i64, serialization_order = 155 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_96226304_2359296"} : tensor<768x768xf32>
    %156 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 96223232 : i64, serialization_order = 156 : i64, size = 3072 : i64, source_name = "torch_hip_weight_96223232_3072"} : tensor<768xf32>
    %157 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 93863936 : i64, serialization_order = 157 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_93863936_2359296"} : tensor<768x768xf32>
    %158 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 93860864 : i64, serialization_order = 158 : i64, size = 3072 : i64, source_name = "torch_hip_weight_93860864_3072"} : tensor<768xf32>
    %159 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 91501568 : i64, serialization_order = 159 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_91501568_2359296"} : tensor<768x768xf32>
    %160 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 91498496 : i64, serialization_order = 160 : i64, size = 3072 : i64, source_name = "torch_hip_weight_91498496_3072"} : tensor<768xf32>
    %161 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 89139200 : i64, serialization_order = 161 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_89139200_2359296"} : tensor<768x768xf32>
    %162 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 89136128 : i64, serialization_order = 162 : i64, size = 3072 : i64, source_name = "torch_hip_weight_89136128_3072"} : tensor<768xf32>
    %163 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 89133056 : i64, serialization_order = 163 : i64, size = 3072 : i64, source_name = "torch_hip_weight_89133056_3072"} : tensor<768xf32>
    %164 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 89129984 : i64, serialization_order = 164 : i64, size = 3072 : i64, source_name = "torch_hip_weight_89129984_3072"} : tensor<768xf32>
    %165 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 79692800 : i64, serialization_order = 165 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_79692800_9437184"} : tensor<768x3072xf32>
    %166 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 79680512 : i64, serialization_order = 166 : i64, size = 12288 : i64, source_name = "torch_hip_weight_79680512_12288"} : tensor<3072xf32>
    %167 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 70243328 : i64, serialization_order = 167 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_70243328_9437184"} : tensor<3072x768xf32>
    %168 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 70240256 : i64, serialization_order = 168 : i64, size = 3072 : i64, source_name = "torch_hip_weight_70240256_3072"} : tensor<768xf32>
    %169 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 70237184 : i64, serialization_order = 169 : i64, size = 3072 : i64, source_name = "torch_hip_weight_70237184_3072"} : tensor<768xf32>
    %170 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 70234112 : i64, serialization_order = 170 : i64, size = 3072 : i64, source_name = "torch_hip_weight_70234112_3072"} : tensor<768xf32>
    %171 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 67874816 : i64, serialization_order = 171 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_67874816_2359296"} : tensor<768x768xf32>
    %172 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 67871744 : i64, serialization_order = 172 : i64, size = 3072 : i64, source_name = "torch_hip_weight_67871744_3072"} : tensor<768xf32>
    %173 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 65512448 : i64, serialization_order = 173 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_65512448_2359296"} : tensor<768x768xf32>
    %174 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 65509376 : i64, serialization_order = 174 : i64, size = 3072 : i64, source_name = "torch_hip_weight_65509376_3072"} : tensor<768xf32>
    %175 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 63150080 : i64, serialization_order = 175 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_63150080_2359296"} : tensor<768x768xf32>
    %176 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 63147008 : i64, serialization_order = 176 : i64, size = 3072 : i64, source_name = "torch_hip_weight_63147008_3072"} : tensor<768xf32>
    %177 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 60787712 : i64, serialization_order = 177 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_60787712_2359296"} : tensor<768x768xf32>
    %178 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 60784640 : i64, serialization_order = 178 : i64, size = 3072 : i64, source_name = "torch_hip_weight_60784640_3072"} : tensor<768xf32>
    %179 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 60781568 : i64, serialization_order = 179 : i64, size = 3072 : i64, source_name = "torch_hip_weight_60781568_3072"} : tensor<768xf32>
    %180 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 60778496 : i64, serialization_order = 180 : i64, size = 3072 : i64, source_name = "torch_hip_weight_60778496_3072"} : tensor<768xf32>
    %181 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 51341312 : i64, serialization_order = 181 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_51341312_9437184"} : tensor<768x3072xf32>
    %182 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 51329024 : i64, serialization_order = 182 : i64, size = 12288 : i64, source_name = "torch_hip_weight_51329024_12288"} : tensor<3072xf32>
    %183 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 41891840 : i64, serialization_order = 183 : i64, size = 9437184 : i64, source_name = "torch_hip_weight_41891840_9437184"} : tensor<3072x768xf32>
    %184 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 41888768 : i64, serialization_order = 184 : i64, size = 3072 : i64, source_name = "torch_hip_weight_41888768_3072"} : tensor<768xf32>
    %185 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 41885696 : i64, serialization_order = 185 : i64, size = 3072 : i64, source_name = "torch_hip_weight_41885696_3072"} : tensor<768xf32>
    %186 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 41882624 : i64, serialization_order = 186 : i64, size = 3072 : i64, source_name = "torch_hip_weight_41882624_3072"} : tensor<768xf32>
    %187 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 39523328 : i64, serialization_order = 187 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_39523328_2359296"} : tensor<768x768xf32>
    %cst = arith.constant 1.250000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %188 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 39520256 : i64, serialization_order = 188 : i64, size = 3072 : i64, source_name = "torch_hip_weight_39520256_3072"} : tensor<768xf32>
    %189 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 37160960 : i64, serialization_order = 189 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_37160960_2359296"} : tensor<768x768xf32>
    %190 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 37157888 : i64, serialization_order = 190 : i64, size = 3072 : i64, source_name = "torch_hip_weight_37157888_3072"} : tensor<768xf32>
    %191 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 34798592 : i64, serialization_order = 191 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_34798592_2359296"} : tensor<768x768xf32>
    %c64_i64 = arith.constant 64 : i64
    %192 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 34795520 : i64, serialization_order = 192 : i64, size = 3072 : i64, source_name = "torch_hip_weight_34795520_3072"} : tensor<768xf32>
    %193 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 32436224 : i64, serialization_order = 193 : i64, size = 2359296 : i64, source_name = "torch_hip_weight_32436224_2359296"} : tensor<768x768xf32>
    %194 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 32433152 : i64, serialization_order = 194 : i64, size = 3072 : i64, source_name = "torch_hip_weight_32433152_3072"} : tensor<768xf32>
    %195 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 32430080 : i64, serialization_order = 195 : i64, size = 3072 : i64, source_name = "torch_hip_weight_32430080_3072"} : tensor<768xf32>
    %196 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 32427008 : i64, serialization_order = 196 : i64, size = 3072 : i64, source_name = "torch_hip_weight_32427008_3072"} : tensor<768xf32>
    %197 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 29625344 : i64, serialization_order = 197 : i64, size = 2801664 : i64, source_name = "torch_hip_weight_29625344_2801664"} : tensor<768x48x19xf32>
    %198 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 29622272 : i64, serialization_order = 198 : i64, size = 3072 : i64, source_name = "torch_hip_weight_29622272_3072"} : tensor<768xf32>
    %199 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 26820608 : i64, serialization_order = 199 : i64, size = 2801664 : i64, source_name = "torch_hip_weight_26820608_2801664"} : tensor<768x48x19xf32>
    %200 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 26817536 : i64, serialization_order = 200 : i64, size = 3072 : i64, source_name = "torch_hip_weight_26817536_3072"} : tensor<768xf32>
    %201 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 24015872 : i64, serialization_order = 201 : i64, size = 2801664 : i64, source_name = "torch_hip_weight_24015872_2801664"} : tensor<768x48x19xf32>
    %202 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 24012800 : i64, serialization_order = 202 : i64, size = 3072 : i64, source_name = "torch_hip_weight_24012800_3072"} : tensor<768xf32>
    %203 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 21211136 : i64, serialization_order = 203 : i64, size = 2801664 : i64, source_name = "torch_hip_weight_21211136_2801664"} : tensor<768x48x19xf32>
    %c768_i64 = arith.constant 768 : i64
    %c16_i64 = arith.constant 16 : i64
    %c9_i64 = arith.constant 9 : i64
    %204 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 21208064 : i64, serialization_order = 204 : i64, size = 3072 : i64, source_name = "torch_hip_weight_21208064_3072"} : tensor<768xf32>
    %205 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 18406400 : i64, serialization_order = 205 : i64, size = 2801664 : i64, source_name = "torch_hip_weight_18406400_2801664"} : tensor<768x48x19xf32>
    %c3_i64 = arith.constant 3 : i64
    %c169_i64 = arith.constant 169 : i64
    %206 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 18403328 : i64, serialization_order = 206 : i64, size = 3072 : i64, source_name = "torch_hip_weight_18403328_3072"} : tensor<768xf32>
    %207 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 16830464 : i64, serialization_order = 207 : i64, size = 1572864 : i64, source_name = "torch_hip_weight_16830464_1572864"} : tensor<768x512xf32>
    %208 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 16828416 : i64, serialization_order = 208 : i64, size = 2048 : i64, source_name = "torch_hip_weight_16828416_2048"} : tensor<512xf32>
    %209 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 16826368 : i64, serialization_order = 209 : i64, size = 2048 : i64, source_name = "torch_hip_weight_16826368_2048"} : tensor<512xf32>
    %210 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 16824320 : i64, serialization_order = 210 : i64, size = 2048 : i64, source_name = "torch_hip_weight_16824320_2048"} : tensor<512xf32>
    %211 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 16822272 : i64, serialization_order = 211 : i64, size = 2048 : i64, source_name = "torch_hip_weight_16822272_2048"} : tensor<512xf32>
    %212 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 14725120 : i64, serialization_order = 212 : i64, size = 2097152 : i64, source_name = "torch_hip_weight_14725120_2097152"} : tensor<512x512x2xf32>
    %213 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 14723072 : i64, serialization_order = 213 : i64, size = 2048 : i64, source_name = "torch_hip_weight_14723072_2048"} : tensor<512xf32>
    %214 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 14721024 : i64, serialization_order = 214 : i64, size = 2048 : i64, source_name = "torch_hip_weight_14721024_2048"} : tensor<512xf32>
    %215 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 12623872 : i64, serialization_order = 215 : i64, size = 2097152 : i64, source_name = "torch_hip_weight_12623872_2097152"} : tensor<512x512x2xf32>
    %216 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 12621824 : i64, serialization_order = 216 : i64, size = 2048 : i64, source_name = "torch_hip_weight_12621824_2048"} : tensor<512xf32>
    %217 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 12619776 : i64, serialization_order = 217 : i64, size = 2048 : i64, source_name = "torch_hip_weight_12619776_2048"} : tensor<512xf32>
    %218 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 9474048 : i64, serialization_order = 218 : i64, size = 3145728 : i64, source_name = "torch_hip_weight_9474048_3145728"} : tensor<512x512x3xf32>
    %219 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 9472000 : i64, serialization_order = 219 : i64, size = 2048 : i64, source_name = "torch_hip_weight_9472000_2048"} : tensor<512xf32>
    %220 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 9469952 : i64, serialization_order = 220 : i64, size = 2048 : i64, source_name = "torch_hip_weight_9469952_2048"} : tensor<512xf32>
    %221 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 6324224 : i64, serialization_order = 221 : i64, size = 3145728 : i64, source_name = "torch_hip_weight_6324224_3145728"} : tensor<512x512x3xf32>
    %222 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 6322176 : i64, serialization_order = 222 : i64, size = 2048 : i64, source_name = "torch_hip_weight_6322176_2048"} : tensor<512xf32>
    %223 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 6320128 : i64, serialization_order = 223 : i64, size = 2048 : i64, source_name = "torch_hip_weight_6320128_2048"} : tensor<512xf32>
    %224 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 3174400 : i64, serialization_order = 224 : i64, size = 3145728 : i64, source_name = "torch_hip_weight_3174400_3145728"} : tensor<512x512x3xf32>
    %225 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 3172352 : i64, serialization_order = 225 : i64, size = 2048 : i64, source_name = "torch_hip_weight_3172352_2048"} : tensor<512xf32>
    %226 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 3170304 : i64, serialization_order = 226 : i64, size = 2048 : i64, source_name = "torch_hip_weight_3170304_2048"} : tensor<512xf32>
    %c2_i64 = arith.constant 2 : i64
    %227 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 24576 : i64, serialization_order = 227 : i64, size = 3145728 : i64, source_name = "torch_hip_weight_24576_3145728"} : tensor<512x512x3xf32>
    %false = arith.constant false
    %cst_1 = arith.constant 1.000000e-05 : f64
    %228 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 22528 : i64, serialization_order = 228 : i64, size = 2048 : i64, source_name = "torch_hip_weight_22528_2048"} : tensor<512xf32>
    %229 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 20480 : i64, serialization_order = 229 : i64, size = 2048 : i64, source_name = "torch_hip_weight_20480_2048"} : tensor<512xf32>
    %c512_i64 = arith.constant 512 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c-2_i64 = arith.constant -2 : i64
    %c0_i64 = arith.constant 0 : i64
    %c5_i64 = arith.constant 5 : i64
    %230 = hip.constant {location = "facebook-data2vec-audio-base-960h.weights.bin", offset = 0 : i64, serialization_order = 230 : i64, size = 20480 : i64, source_name = "torch_hip_weight_0_20480"} : tensor<512x1x10xf32>
    %c1_i64 = arith.constant 1 : i64
    %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] output_shape [1, 1, 54400] : tensor<1x54400xf32> into tensor<1x1x54400xf32>
    %cst_2 = arith.constant dense<5> : tensor<1xi64>
    %cst_3 = arith.constant dense<0> : tensor<1xi64>
    %cst_4 = arith.constant dense<1> : tensor<1xi64>
    %cst_5 = arith.constant dense<> : tensor<0xi64>
    %231 = tensor.empty() : tensor<1x512x10879xf32>
    %232 = hip.conv(%arg0) ins(%expanded, %230 : tensor<1x1x54400xf32>, tensor<512x1x10xf32>) outs(%231 : tensor<1x512x10879xf32>) {dilations = [1], group = 1 : i64, kernel_shape = [10], pads = [0, 0], strides = [5]} : tensor<1x512x10879xf32>
    %233 = tensor.empty() : tensor<1x10879x512xf32>
    %234 = hip.transpose(%arg0) ins(%232 : tensor<1x512x10879xf32>) outs(%233 : tensor<1x10879x512xf32>) {perm = [0, 2, 1]} : tensor<1x10879x512xf32>
    %cst_6 = arith.constant dense<512> : tensor<1xi64>
    %235 = tensor.empty() : tensor<1x10879x512xf32>
    %236 = hip.layer_norm(%arg0) ins(%234, %229, %228 : tensor<1x10879x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%235 : tensor<1x10879x512xf32>) {axis = 2 : i64} : tensor<1x10879x512xf32>
    %237 = tensor.empty() : tensor<1x512x10879xf32>
    %238 = hip.transpose(%arg0) ins(%236 : tensor<1x10879x512xf32>) outs(%237 : tensor<1x512x10879xf32>) {perm = [0, 2, 1]} : tensor<1x512x10879xf32>
    %239 = tensor.empty() : tensor<1x512x10879xf32>
    %240 = hip.gelu(%arg0) ins(%238 : tensor<1x512x10879xf32>) outs(%239 : tensor<1x512x10879xf32>) : tensor<1x512x10879xf32>
    %cst_7 = arith.constant dense<2> : tensor<1xi64>
    %cst_8 = arith.constant dense<0> : tensor<1xi64>
    %cst_9 = arith.constant dense<1> : tensor<1xi64>
    %cst_10 = arith.constant dense<> : tensor<0xi64>
    %241 = tensor.empty() : tensor<1x512x5439xf32>
    %242 = hip.conv(%arg0) ins(%240, %227 : tensor<1x512x10879xf32>, tensor<512x512x3xf32>) outs(%241 : tensor<1x512x5439xf32>) {dilations = [1], group = 1 : i64, kernel_shape = [3], pads = [0, 0], strides = [2]} : tensor<1x512x5439xf32>
    %243 = tensor.empty() : tensor<1x5439x512xf32>
    %244 = hip.transpose(%arg0) ins(%242 : tensor<1x512x5439xf32>) outs(%243 : tensor<1x5439x512xf32>) {perm = [0, 2, 1]} : tensor<1x5439x512xf32>
    %cst_11 = arith.constant dense<512> : tensor<1xi64>
    %245 = tensor.empty() : tensor<1x5439x512xf32>
    %246 = hip.layer_norm(%arg0) ins(%244, %226, %225 : tensor<1x5439x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%245 : tensor<1x5439x512xf32>) {axis = 2 : i64} : tensor<1x5439x512xf32>
    %247 = tensor.empty() : tensor<1x512x5439xf32>
    %248 = hip.transpose(%arg0) ins(%246 : tensor<1x5439x512xf32>) outs(%247 : tensor<1x512x5439xf32>) {perm = [0, 2, 1]} : tensor<1x512x5439xf32>
    %249 = tensor.empty() : tensor<1x512x5439xf32>
    %250 = hip.gelu(%arg0) ins(%248 : tensor<1x512x5439xf32>) outs(%249 : tensor<1x512x5439xf32>) : tensor<1x512x5439xf32>
    %cst_12 = arith.constant dense<2> : tensor<1xi64>
    %cst_13 = arith.constant dense<0> : tensor<1xi64>
    %cst_14 = arith.constant dense<1> : tensor<1xi64>
    %cst_15 = arith.constant dense<> : tensor<0xi64>
    %251 = tensor.empty() : tensor<1x512x2719xf32>
    %252 = hip.conv(%arg0) ins(%250, %224 : tensor<1x512x5439xf32>, tensor<512x512x3xf32>) outs(%251 : tensor<1x512x2719xf32>) {dilations = [1], group = 1 : i64, kernel_shape = [3], pads = [0, 0], strides = [2]} : tensor<1x512x2719xf32>
    %253 = tensor.empty() : tensor<1x2719x512xf32>
    %254 = hip.transpose(%arg0) ins(%252 : tensor<1x512x2719xf32>) outs(%253 : tensor<1x2719x512xf32>) {perm = [0, 2, 1]} : tensor<1x2719x512xf32>
    %cst_16 = arith.constant dense<512> : tensor<1xi64>
    %255 = tensor.empty() : tensor<1x2719x512xf32>
    %256 = hip.layer_norm(%arg0) ins(%254, %223, %222 : tensor<1x2719x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%255 : tensor<1x2719x512xf32>) {axis = 2 : i64} : tensor<1x2719x512xf32>
    %257 = tensor.empty() : tensor<1x512x2719xf32>
    %258 = hip.transpose(%arg0) ins(%256 : tensor<1x2719x512xf32>) outs(%257 : tensor<1x512x2719xf32>) {perm = [0, 2, 1]} : tensor<1x512x2719xf32>
    %259 = tensor.empty() : tensor<1x512x2719xf32>
    %260 = hip.gelu(%arg0) ins(%258 : tensor<1x512x2719xf32>) outs(%259 : tensor<1x512x2719xf32>) : tensor<1x512x2719xf32>
    %cst_17 = arith.constant dense<2> : tensor<1xi64>
    %cst_18 = arith.constant dense<0> : tensor<1xi64>
    %cst_19 = arith.constant dense<1> : tensor<1xi64>
    %cst_20 = arith.constant dense<> : tensor<0xi64>
    %261 = tensor.empty() : tensor<1x512x1359xf32>
    %262 = hip.conv(%arg0) ins(%260, %221 : tensor<1x512x2719xf32>, tensor<512x512x3xf32>) outs(%261 : tensor<1x512x1359xf32>) {dilations = [1], group = 1 : i64, kernel_shape = [3], pads = [0, 0], strides = [2]} : tensor<1x512x1359xf32>
    %263 = tensor.empty() : tensor<1x1359x512xf32>
    %264 = hip.transpose(%arg0) ins(%262 : tensor<1x512x1359xf32>) outs(%263 : tensor<1x1359x512xf32>) {perm = [0, 2, 1]} : tensor<1x1359x512xf32>
    %cst_21 = arith.constant dense<512> : tensor<1xi64>
    %265 = tensor.empty() : tensor<1x1359x512xf32>
    %266 = hip.layer_norm(%arg0) ins(%264, %220, %219 : tensor<1x1359x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%265 : tensor<1x1359x512xf32>) {axis = 2 : i64} : tensor<1x1359x512xf32>
    %267 = tensor.empty() : tensor<1x512x1359xf32>
    %268 = hip.transpose(%arg0) ins(%266 : tensor<1x1359x512xf32>) outs(%267 : tensor<1x512x1359xf32>) {perm = [0, 2, 1]} : tensor<1x512x1359xf32>
    %269 = tensor.empty() : tensor<1x512x1359xf32>
    %270 = hip.gelu(%arg0) ins(%268 : tensor<1x512x1359xf32>) outs(%269 : tensor<1x512x1359xf32>) : tensor<1x512x1359xf32>
    %cst_22 = arith.constant dense<2> : tensor<1xi64>
    %cst_23 = arith.constant dense<0> : tensor<1xi64>
    %cst_24 = arith.constant dense<1> : tensor<1xi64>
    %cst_25 = arith.constant dense<> : tensor<0xi64>
    %271 = tensor.empty() : tensor<1x512x679xf32>
    %272 = hip.conv(%arg0) ins(%270, %218 : tensor<1x512x1359xf32>, tensor<512x512x3xf32>) outs(%271 : tensor<1x512x679xf32>) {dilations = [1], group = 1 : i64, kernel_shape = [3], pads = [0, 0], strides = [2]} : tensor<1x512x679xf32>
    %273 = tensor.empty() : tensor<1x679x512xf32>
    %274 = hip.transpose(%arg0) ins(%272 : tensor<1x512x679xf32>) outs(%273 : tensor<1x679x512xf32>) {perm = [0, 2, 1]} : tensor<1x679x512xf32>
    %cst_26 = arith.constant dense<512> : tensor<1xi64>
    %275 = tensor.empty() : tensor<1x679x512xf32>
    %276 = hip.layer_norm(%arg0) ins(%274, %217, %216 : tensor<1x679x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%275 : tensor<1x679x512xf32>) {axis = 2 : i64} : tensor<1x679x512xf32>
    %277 = tensor.empty() : tensor<1x512x679xf32>
    %278 = hip.transpose(%arg0) ins(%276 : tensor<1x679x512xf32>) outs(%277 : tensor<1x512x679xf32>) {perm = [0, 2, 1]} : tensor<1x512x679xf32>
    %279 = tensor.empty() : tensor<1x512x679xf32>
    %280 = hip.gelu(%arg0) ins(%278 : tensor<1x512x679xf32>) outs(%279 : tensor<1x512x679xf32>) : tensor<1x512x679xf32>
    %cst_27 = arith.constant dense<2> : tensor<1xi64>
    %cst_28 = arith.constant dense<0> : tensor<1xi64>
    %cst_29 = arith.constant dense<1> : tensor<1xi64>
    %cst_30 = arith.constant dense<> : tensor<0xi64>
    %281 = tensor.empty() : tensor<1x512x339xf32>
    %282 = hip.conv(%arg0) ins(%280, %215 : tensor<1x512x679xf32>, tensor<512x512x2xf32>) outs(%281 : tensor<1x512x339xf32>) {dilations = [1], group = 1 : i64, kernel_shape = [2], pads = [0, 0], strides = [2]} : tensor<1x512x339xf32>
    %283 = tensor.empty() : tensor<1x339x512xf32>
    %284 = hip.transpose(%arg0) ins(%282 : tensor<1x512x339xf32>) outs(%283 : tensor<1x339x512xf32>) {perm = [0, 2, 1]} : tensor<1x339x512xf32>
    %cst_31 = arith.constant dense<512> : tensor<1xi64>
    %285 = tensor.empty() : tensor<1x339x512xf32>
    %286 = hip.layer_norm(%arg0) ins(%284, %214, %213 : tensor<1x339x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%285 : tensor<1x339x512xf32>) {axis = 2 : i64} : tensor<1x339x512xf32>
    %287 = tensor.empty() : tensor<1x512x339xf32>
    %288 = hip.transpose(%arg0) ins(%286 : tensor<1x339x512xf32>) outs(%287 : tensor<1x512x339xf32>) {perm = [0, 2, 1]} : tensor<1x512x339xf32>
    %289 = tensor.empty() : tensor<1x512x339xf32>
    %290 = hip.gelu(%arg0) ins(%288 : tensor<1x512x339xf32>) outs(%289 : tensor<1x512x339xf32>) : tensor<1x512x339xf32>
    %cst_32 = arith.constant dense<2> : tensor<1xi64>
    %cst_33 = arith.constant dense<0> : tensor<1xi64>
    %cst_34 = arith.constant dense<1> : tensor<1xi64>
    %cst_35 = arith.constant dense<> : tensor<0xi64>
    %291 = tensor.empty() : tensor<1x512x169xf32>
    %292 = hip.conv(%arg0) ins(%290, %212 : tensor<1x512x339xf32>, tensor<512x512x2xf32>) outs(%291 : tensor<1x512x169xf32>) {dilations = [1], group = 1 : i64, kernel_shape = [2], pads = [0, 0], strides = [2]} : tensor<1x512x169xf32>
    %293 = tensor.empty() : tensor<1x169x512xf32>
    %294 = hip.transpose(%arg0) ins(%292 : tensor<1x512x169xf32>) outs(%293 : tensor<1x169x512xf32>) {perm = [0, 2, 1]} : tensor<1x169x512xf32>
    %cst_36 = arith.constant dense<512> : tensor<1xi64>
    %295 = tensor.empty() : tensor<1x169x512xf32>
    %296 = hip.layer_norm(%arg0) ins(%294, %211, %210 : tensor<1x169x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%295 : tensor<1x169x512xf32>) {axis = 2 : i64} : tensor<1x169x512xf32>
    %297 = tensor.empty() : tensor<1x512x169xf32>
    %298 = hip.transpose(%arg0) ins(%296 : tensor<1x169x512xf32>) outs(%297 : tensor<1x512x169xf32>) {perm = [0, 2, 1]} : tensor<1x512x169xf32>
    %299 = tensor.empty() : tensor<1x512x169xf32>
    %300 = hip.gelu(%arg0) ins(%298 : tensor<1x512x169xf32>) outs(%299 : tensor<1x512x169xf32>) : tensor<1x512x169xf32>
    %301 = tensor.empty() : tensor<1x169x512xf32>
    %302 = hip.transpose(%arg0) ins(%300 : tensor<1x512x169xf32>) outs(%301 : tensor<1x169x512xf32>) {perm = [0, 2, 1]} : tensor<1x169x512xf32>
    %cst_37 = arith.constant dense<512> : tensor<1xi64>
    %303 = tensor.empty() : tensor<1x169x512xf32>
    %304 = hip.layer_norm(%arg0) ins(%302, %209, %208 : tensor<1x169x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%303 : tensor<1x169x512xf32>) {axis = 2 : i64} : tensor<1x169x512xf32>
    %cst_38 = arith.constant dense<[169, 512]> : tensor<2xi64>
    %reshape = tensor.reshape %304(%cst_38) : (tensor<1x169x512xf32>, tensor<2xi64>) -> tensor<169x512xf32>
    %305 = tensor.empty() : tensor<169x768xf32>
    %306 = hip.gemm(%arg0) ins(%reshape, %207, %206 : tensor<169x512xf32>, tensor<768x512xf32>, tensor<768xf32>) outs(%305 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_39 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_40 = tensor.reshape %306(%cst_39) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %307 = hip.constant {serialization_order = 231 : i64, value = dense<0> : tensor<i64>} : tensor<i64>
    %splat = tensor.splat %c0_i64 : tensor<i64>
    %308 = hip.constant {serialization_order = 232 : i64, value = dense<169> : tensor<i64>} : tensor<i64>
    %splat_41 = tensor.splat %c169_i64 : tensor<i64>
    %309 = hip.constant {serialization_order = 233 : i64, value = dense<1> : tensor<i64>} : tensor<i64>
    %splat_42 = tensor.splat %c1_i64 : tensor<i64>
    %310 = tensor.empty() : tensor<169xi64>
    %311 = hip.range(%arg0) ins(%307, %308, %309 : tensor<i64>, tensor<i64>, tensor<i64>) outs(%310 : tensor<169xi64>) : tensor<169xi64>
    %expanded_43 = tensor.expand_shape %311 [[0, 1]] output_shape [1, 169] : tensor<169xi64> into tensor<1x169xi64>
    %expanded_44 = tensor.expand_shape %expanded_43 [[0, 1], [2]] output_shape [1, 1, 169] : tensor<1x169xi64> into tensor<1x1x169xi64>
    %expanded_45 = tensor.expand_shape %expanded_44 [[0], [1], [2, 3]] output_shape [1, 1, 169, 1] : tensor<1x1x169xi64> into tensor<1x1x169x1xi64>
    %312 = hip.constant {serialization_order = 234 : i64, value = dense<0> : tensor<i64>} : tensor<i64>
    %splat_46 = tensor.splat %c0_i64 : tensor<i64>
    %313 = tensor.empty() : tensor<1x1x169x1xi1>
    %314 = tensor.empty() : tensor<1x1x169x1xi1>
    %315 = hip.less(%arg0) ins(%expanded_45, %312 : tensor<1x1x169x1xi64>, tensor<i64>) outs(%313 : tensor<1x1x169x1xi1>) : tensor<1x1x169x1xi1>
    %316 = hip.not(%arg0) ins(%315 : tensor<1x1x169x1xi1>) outs(%314 : tensor<1x1x169x1xi1>) : tensor<1x1x169x1xi1>
    %cst_47 = arith.constant dense<[1, -1, 169, 169]> : tensor<4xi64>
    %317 = tensor.empty() : tensor<1x1x169x169xi1>
    %318 = hip.constant {serialization_order = 235 : i64, value = dense<[1, 1, 169, 169]> : tensor<4xi64>} : tensor<4xi64>
    %cst_48 = arith.constant dense<[1, 1, 169, 169]> : tensor<4xi64>
    %319 = hip.expand(%arg0) ins(%316, %318 : tensor<1x1x169x1xi1>, tensor<4xi64>) outs(%317 : tensor<1x1x169x169xi1>) : tensor<1x1x169x169xi1>
    %320 = tensor.empty() : tensor<1x768x169xf32>
    %321 = hip.transpose(%arg0) ins(%reshape_40 : tensor<1x169x768xf32>) outs(%320 : tensor<1x768x169xf32>) {perm = [0, 2, 1]} : tensor<1x768x169xf32>
    %cst_49 = arith.constant dense<1> : tensor<1xi64>
    %cst_50 = arith.constant dense<9> : tensor<1xi64>
    %cst_51 = arith.constant dense<1> : tensor<1xi64>
    %cst_52 = arith.constant dense<> : tensor<0xi64>
    %322 = tensor.empty() : tensor<1x768x169xf32>
    %323 = hip.conv(%arg0) ins(%321, %205, %204 : tensor<1x768x169xf32>, tensor<768x48x19xf32>, tensor<768xf32>) outs(%322 : tensor<1x768x169xf32>) {dilations = [1], group = 16 : i64, kernel_shape = [19], pads = [9, 9], strides = [1]} : tensor<1x768x169xf32>
    %324 = tensor.empty() : tensor<1x169x768xf32>
    %325 = hip.transpose(%arg0) ins(%323 : tensor<1x768x169xf32>) outs(%324 : tensor<1x169x768xf32>) {perm = [0, 2, 1]} : tensor<1x169x768xf32>
    %cst_53 = arith.constant dense<768> : tensor<1xi64>
    %326 = hip.constant {serialization_order = 236 : i64, value = dense<1.000000e+00> : tensor<768xf32>} : tensor<768xf32>
    %cst_54 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %327 = tensor.empty() : tensor<1x169x768xf32>
    %328 = hip.layer_norm(%arg0) ins(%325, %326 : tensor<1x169x768xf32>, tensor<768xf32>) outs(%327 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %329 = tensor.empty() : tensor<1x768x169xf32>
    %330 = hip.transpose(%arg0) ins(%328 : tensor<1x169x768xf32>) outs(%329 : tensor<1x768x169xf32>) {perm = [0, 2, 1]} : tensor<1x768x169xf32>
    %331 = tensor.empty() : tensor<1x768x169xf32>
    %332 = hip.gelu(%arg0) ins(%330 : tensor<1x768x169xf32>) outs(%331 : tensor<1x768x169xf32>) : tensor<1x768x169xf32>
    %cst_55 = arith.constant dense<1> : tensor<1xi64>
    %cst_56 = arith.constant dense<9> : tensor<1xi64>
    %cst_57 = arith.constant dense<1> : tensor<1xi64>
    %cst_58 = arith.constant dense<> : tensor<0xi64>
    %333 = tensor.empty() : tensor<1x768x169xf32>
    %334 = hip.conv(%arg0) ins(%332, %203, %202 : tensor<1x768x169xf32>, tensor<768x48x19xf32>, tensor<768xf32>) outs(%333 : tensor<1x768x169xf32>) {dilations = [1], group = 16 : i64, kernel_shape = [19], pads = [9, 9], strides = [1]} : tensor<1x768x169xf32>
    %335 = tensor.empty() : tensor<1x169x768xf32>
    %336 = hip.transpose(%arg0) ins(%334 : tensor<1x768x169xf32>) outs(%335 : tensor<1x169x768xf32>) {perm = [0, 2, 1]} : tensor<1x169x768xf32>
    %cst_59 = arith.constant dense<768> : tensor<1xi64>
    %337 = hip.constant {serialization_order = 237 : i64, value = dense<1.000000e+00> : tensor<768xf32>} : tensor<768xf32>
    %cst_60 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %338 = tensor.empty() : tensor<1x169x768xf32>
    %339 = hip.layer_norm(%arg0) ins(%336, %337 : tensor<1x169x768xf32>, tensor<768xf32>) outs(%338 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %340 = tensor.empty() : tensor<1x768x169xf32>
    %341 = hip.transpose(%arg0) ins(%339 : tensor<1x169x768xf32>) outs(%340 : tensor<1x768x169xf32>) {perm = [0, 2, 1]} : tensor<1x768x169xf32>
    %342 = tensor.empty() : tensor<1x768x169xf32>
    %343 = hip.gelu(%arg0) ins(%341 : tensor<1x768x169xf32>) outs(%342 : tensor<1x768x169xf32>) : tensor<1x768x169xf32>
    %cst_61 = arith.constant dense<1> : tensor<1xi64>
    %cst_62 = arith.constant dense<9> : tensor<1xi64>
    %cst_63 = arith.constant dense<1> : tensor<1xi64>
    %cst_64 = arith.constant dense<> : tensor<0xi64>
    %344 = tensor.empty() : tensor<1x768x169xf32>
    %345 = hip.conv(%arg0) ins(%343, %201, %200 : tensor<1x768x169xf32>, tensor<768x48x19xf32>, tensor<768xf32>) outs(%344 : tensor<1x768x169xf32>) {dilations = [1], group = 16 : i64, kernel_shape = [19], pads = [9, 9], strides = [1]} : tensor<1x768x169xf32>
    %346 = tensor.empty() : tensor<1x169x768xf32>
    %347 = hip.transpose(%arg0) ins(%345 : tensor<1x768x169xf32>) outs(%346 : tensor<1x169x768xf32>) {perm = [0, 2, 1]} : tensor<1x169x768xf32>
    %cst_65 = arith.constant dense<768> : tensor<1xi64>
    %348 = hip.constant {serialization_order = 238 : i64, value = dense<1.000000e+00> : tensor<768xf32>} : tensor<768xf32>
    %cst_66 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %349 = tensor.empty() : tensor<1x169x768xf32>
    %350 = hip.layer_norm(%arg0) ins(%347, %348 : tensor<1x169x768xf32>, tensor<768xf32>) outs(%349 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %351 = tensor.empty() : tensor<1x768x169xf32>
    %352 = hip.transpose(%arg0) ins(%350 : tensor<1x169x768xf32>) outs(%351 : tensor<1x768x169xf32>) {perm = [0, 2, 1]} : tensor<1x768x169xf32>
    %353 = tensor.empty() : tensor<1x768x169xf32>
    %354 = hip.gelu(%arg0) ins(%352 : tensor<1x768x169xf32>) outs(%353 : tensor<1x768x169xf32>) : tensor<1x768x169xf32>
    %cst_67 = arith.constant dense<1> : tensor<1xi64>
    %cst_68 = arith.constant dense<9> : tensor<1xi64>
    %cst_69 = arith.constant dense<1> : tensor<1xi64>
    %cst_70 = arith.constant dense<> : tensor<0xi64>
    %355 = tensor.empty() : tensor<1x768x169xf32>
    %356 = hip.conv(%arg0) ins(%354, %199, %198 : tensor<1x768x169xf32>, tensor<768x48x19xf32>, tensor<768xf32>) outs(%355 : tensor<1x768x169xf32>) {dilations = [1], group = 16 : i64, kernel_shape = [19], pads = [9, 9], strides = [1]} : tensor<1x768x169xf32>
    %357 = tensor.empty() : tensor<1x169x768xf32>
    %358 = hip.transpose(%arg0) ins(%356 : tensor<1x768x169xf32>) outs(%357 : tensor<1x169x768xf32>) {perm = [0, 2, 1]} : tensor<1x169x768xf32>
    %cst_71 = arith.constant dense<768> : tensor<1xi64>
    %359 = hip.constant {serialization_order = 239 : i64, value = dense<1.000000e+00> : tensor<768xf32>} : tensor<768xf32>
    %cst_72 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %360 = tensor.empty() : tensor<1x169x768xf32>
    %361 = hip.layer_norm(%arg0) ins(%358, %359 : tensor<1x169x768xf32>, tensor<768xf32>) outs(%360 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %362 = tensor.empty() : tensor<1x768x169xf32>
    %363 = hip.transpose(%arg0) ins(%361 : tensor<1x169x768xf32>) outs(%362 : tensor<1x768x169xf32>) {perm = [0, 2, 1]} : tensor<1x768x169xf32>
    %364 = tensor.empty() : tensor<1x768x169xf32>
    %365 = hip.gelu(%arg0) ins(%363 : tensor<1x768x169xf32>) outs(%364 : tensor<1x768x169xf32>) : tensor<1x768x169xf32>
    %cst_73 = arith.constant dense<1> : tensor<1xi64>
    %cst_74 = arith.constant dense<9> : tensor<1xi64>
    %cst_75 = arith.constant dense<1> : tensor<1xi64>
    %cst_76 = arith.constant dense<> : tensor<0xi64>
    %366 = tensor.empty() : tensor<1x768x169xf32>
    %367 = hip.conv(%arg0) ins(%365, %197, %196 : tensor<1x768x169xf32>, tensor<768x48x19xf32>, tensor<768xf32>) outs(%366 : tensor<1x768x169xf32>) {dilations = [1], group = 16 : i64, kernel_shape = [19], pads = [9, 9], strides = [1]} : tensor<1x768x169xf32>
    %368 = tensor.empty() : tensor<1x169x768xf32>
    %369 = hip.transpose(%arg0) ins(%367 : tensor<1x768x169xf32>) outs(%368 : tensor<1x169x768xf32>) {perm = [0, 2, 1]} : tensor<1x169x768xf32>
    %cst_77 = arith.constant dense<768> : tensor<1xi64>
    %370 = hip.constant {serialization_order = 240 : i64, value = dense<1.000000e+00> : tensor<768xf32>} : tensor<768xf32>
    %cst_78 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %371 = tensor.empty() : tensor<1x169x768xf32>
    %372 = hip.layer_norm(%arg0) ins(%369, %370 : tensor<1x169x768xf32>, tensor<768xf32>) outs(%371 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %373 = tensor.empty() : tensor<1x768x169xf32>
    %374 = hip.transpose(%arg0) ins(%372 : tensor<1x169x768xf32>) outs(%373 : tensor<1x768x169xf32>) {perm = [0, 2, 1]} : tensor<1x768x169xf32>
    %375 = tensor.empty() : tensor<1x768x169xf32>
    %376 = hip.gelu(%arg0) ins(%374 : tensor<1x768x169xf32>) outs(%375 : tensor<1x768x169xf32>) : tensor<1x768x169xf32>
    %377 = tensor.empty() : tensor<1x169x768xf32>
    %378 = hip.transpose(%arg0) ins(%376 : tensor<1x768x169xf32>) outs(%377 : tensor<1x169x768xf32>) {perm = [0, 2, 1]} : tensor<1x169x768xf32>
    %379 = tensor.empty() : tensor<1x169x768xf32>
    %380 = hip.add(%arg0) ins(%reshape_40, %378 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%379 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_79 = arith.constant dense<768> : tensor<1xi64>
    %381 = tensor.empty() : tensor<1x169x768xf32>
    %382 = hip.layer_norm(%arg0) ins(%380, %195, %194 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%381 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_80 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_81 = tensor.reshape %382(%cst_80) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %383 = tensor.empty() : tensor<169x768xf32>
    %384 = hip.gemm(%arg0) ins(%reshape_81, %193, %192 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%383 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_82 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_83 = tensor.reshape %384(%cst_82) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_84 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_85 = tensor.reshape %reshape_83(%cst_84) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %385 = tensor.empty() : tensor<1x12x169x64xf32>
    %386 = hip.transpose(%arg0) ins(%reshape_85 : tensor<1x169x12x64xf32>) outs(%385 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_86 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_87 = tensor.reshape %382(%cst_86) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %387 = tensor.empty() : tensor<169x768xf32>
    %388 = hip.gemm(%arg0) ins(%reshape_87, %191, %190 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%387 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_88 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_89 = tensor.reshape %388(%cst_88) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_90 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_91 = tensor.reshape %reshape_89(%cst_90) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %389 = tensor.empty() : tensor<1x12x169x64xf32>
    %390 = hip.transpose(%arg0) ins(%reshape_91 : tensor<1x169x12x64xf32>) outs(%389 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_92 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_93 = tensor.reshape %382(%cst_92) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %391 = tensor.empty() : tensor<169x768xf32>
    %392 = hip.gemm(%arg0) ins(%reshape_93, %189, %188 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%391 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_94 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_95 = tensor.reshape %392(%cst_94) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_96 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_97 = tensor.reshape %reshape_95(%cst_96) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %393 = tensor.empty() : tensor<1x12x169x64xf32>
    %394 = hip.transpose(%arg0) ins(%reshape_97 : tensor<1x169x12x64xf32>) outs(%393 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_98 = arith.constant 0.000000e+00 : f32
    %cst_99 = arith.constant 0xFF800000 : f32
    %395 = hip.constant {serialization_order = 241 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_100 = tensor.splat %cst_98 : tensor<1x1x169x169xf32>
    %396 = hip.constant {serialization_order = 242 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_101 = tensor.splat %cst_99 : tensor<1x1x169x169xf32>
    %397 = tensor.empty() : tensor<1x1x169x169xf32>
    %398 = hip.where(%arg0) ins(%319, %395, %396 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%397 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %399 = tensor.empty() : tensor<1x169x12x64xf32>
    %400 = hip.transpose(%arg0) ins(%386 : tensor<1x12x169x64xf32>) outs(%399 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed = tensor.collapse_shape %400 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %401 = tensor.empty() : tensor<1x169x12x64xf32>
    %402 = hip.transpose(%arg0) ins(%390 : tensor<1x12x169x64xf32>) outs(%401 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_102 = tensor.collapse_shape %402 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %403 = tensor.empty() : tensor<1x169x12x64xf32>
    %404 = hip.transpose(%arg0) ins(%394 : tensor<1x12x169x64xf32>) outs(%403 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_103 = tensor.collapse_shape %404 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %405 = hip.constant {serialization_order = 243 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_104 = arith.constant dense<168> : tensor<1xi32>
    %406 = hip.constant {serialization_order = 244 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_105 = arith.constant dense<169> : tensor<i32>
    %407 = tensor.empty() : tensor<1x169x768xf32>
    %408 = tensor.empty() : tensor<1x12x169x64xf32>
    %409 = tensor.empty() : tensor<1x12x169x64xf32>
    %410:3 = hip.gqa(%arg0) ins(%collapsed, %collapsed_102, %collapsed_103, %405, %406, %398 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%407, %408, %409 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_106 = tensor.expand_shape %410#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %411 = tensor.empty() : tensor<1x12x169x64xf32>
    %412 = hip.transpose(%arg0) ins(%expanded_106 : tensor<1x169x12x64xf32>) outs(%411 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %413 = tensor.empty() : tensor<1x169x12x64xf32>
    %414 = hip.transpose(%arg0) ins(%412 : tensor<1x12x169x64xf32>) outs(%413 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_107 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_108 = tensor.reshape %414(%cst_107) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_109 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_110 = tensor.reshape %reshape_108(%cst_109) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %415 = tensor.empty() : tensor<169x768xf32>
    %416 = hip.gemm(%arg0) ins(%reshape_110, %187, %186 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%415 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_111 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_112 = tensor.reshape %416(%cst_111) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %417 = tensor.empty() : tensor<1x169x768xf32>
    %418 = hip.add(%arg0) ins(%382, %reshape_112 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%417 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_113 = arith.constant dense<768> : tensor<1xi64>
    %419 = tensor.empty() : tensor<1x169x768xf32>
    %420 = hip.layer_norm(%arg0) ins(%418, %185, %184 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%419 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_114 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_115 = tensor.reshape %420(%cst_114) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %421 = tensor.empty() : tensor<169x3072xf32>
    %422 = hip.gemm(%arg0) ins(%reshape_115, %183, %182 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%421 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_116 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_117 = tensor.reshape %422(%cst_116) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %423 = tensor.empty() : tensor<1x169x3072xf32>
    %424 = hip.gelu(%arg0) ins(%reshape_117 : tensor<1x169x3072xf32>) outs(%423 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_118 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_119 = tensor.reshape %424(%cst_118) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %425 = tensor.empty() : tensor<169x768xf32>
    %426 = hip.gemm(%arg0) ins(%reshape_119, %181, %180 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%425 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_120 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_121 = tensor.reshape %426(%cst_120) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %427 = tensor.empty() : tensor<1x169x768xf32>
    %428 = hip.add(%arg0) ins(%420, %reshape_121 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%427 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_122 = arith.constant dense<768> : tensor<1xi64>
    %429 = tensor.empty() : tensor<1x169x768xf32>
    %430 = hip.layer_norm(%arg0) ins(%428, %179, %178 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%429 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_123 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_124 = tensor.reshape %430(%cst_123) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %431 = tensor.empty() : tensor<169x768xf32>
    %432 = hip.gemm(%arg0) ins(%reshape_124, %177, %176 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%431 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_125 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_126 = tensor.reshape %432(%cst_125) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_127 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_128 = tensor.reshape %reshape_126(%cst_127) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %433 = tensor.empty() : tensor<1x12x169x64xf32>
    %434 = hip.transpose(%arg0) ins(%reshape_128 : tensor<1x169x12x64xf32>) outs(%433 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_129 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_130 = tensor.reshape %430(%cst_129) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %435 = tensor.empty() : tensor<169x768xf32>
    %436 = hip.gemm(%arg0) ins(%reshape_130, %175, %174 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%435 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_131 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_132 = tensor.reshape %436(%cst_131) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_133 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_134 = tensor.reshape %reshape_132(%cst_133) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %437 = tensor.empty() : tensor<1x12x169x64xf32>
    %438 = hip.transpose(%arg0) ins(%reshape_134 : tensor<1x169x12x64xf32>) outs(%437 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_135 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_136 = tensor.reshape %430(%cst_135) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %439 = tensor.empty() : tensor<169x768xf32>
    %440 = hip.gemm(%arg0) ins(%reshape_136, %173, %172 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%439 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_137 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_138 = tensor.reshape %440(%cst_137) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_139 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_140 = tensor.reshape %reshape_138(%cst_139) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %441 = tensor.empty() : tensor<1x12x169x64xf32>
    %442 = hip.transpose(%arg0) ins(%reshape_140 : tensor<1x169x12x64xf32>) outs(%441 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_141 = arith.constant 0.000000e+00 : f32
    %cst_142 = arith.constant 0xFF800000 : f32
    %443 = hip.constant {serialization_order = 245 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_143 = tensor.splat %cst_141 : tensor<1x1x169x169xf32>
    %444 = hip.constant {serialization_order = 246 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_144 = tensor.splat %cst_142 : tensor<1x1x169x169xf32>
    %445 = tensor.empty() : tensor<1x1x169x169xf32>
    %446 = hip.where(%arg0) ins(%319, %443, %444 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%445 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %447 = tensor.empty() : tensor<1x169x12x64xf32>
    %448 = hip.transpose(%arg0) ins(%434 : tensor<1x12x169x64xf32>) outs(%447 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_145 = tensor.collapse_shape %448 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %449 = tensor.empty() : tensor<1x169x12x64xf32>
    %450 = hip.transpose(%arg0) ins(%438 : tensor<1x12x169x64xf32>) outs(%449 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_146 = tensor.collapse_shape %450 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %451 = tensor.empty() : tensor<1x169x12x64xf32>
    %452 = hip.transpose(%arg0) ins(%442 : tensor<1x12x169x64xf32>) outs(%451 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_147 = tensor.collapse_shape %452 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %453 = hip.constant {serialization_order = 247 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_148 = arith.constant dense<168> : tensor<1xi32>
    %454 = hip.constant {serialization_order = 248 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_149 = arith.constant dense<169> : tensor<i32>
    %455 = tensor.empty() : tensor<1x169x768xf32>
    %456 = tensor.empty() : tensor<1x12x169x64xf32>
    %457 = tensor.empty() : tensor<1x12x169x64xf32>
    %458:3 = hip.gqa(%arg0) ins(%collapsed_145, %collapsed_146, %collapsed_147, %453, %454, %446 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%455, %456, %457 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_150 = tensor.expand_shape %458#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %459 = tensor.empty() : tensor<1x12x169x64xf32>
    %460 = hip.transpose(%arg0) ins(%expanded_150 : tensor<1x169x12x64xf32>) outs(%459 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %461 = tensor.empty() : tensor<1x169x12x64xf32>
    %462 = hip.transpose(%arg0) ins(%460 : tensor<1x12x169x64xf32>) outs(%461 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_151 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_152 = tensor.reshape %462(%cst_151) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_153 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_154 = tensor.reshape %reshape_152(%cst_153) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %463 = tensor.empty() : tensor<169x768xf32>
    %464 = hip.gemm(%arg0) ins(%reshape_154, %171, %170 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%463 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_155 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_156 = tensor.reshape %464(%cst_155) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %465 = tensor.empty() : tensor<1x169x768xf32>
    %466 = hip.add(%arg0) ins(%430, %reshape_156 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%465 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_157 = arith.constant dense<768> : tensor<1xi64>
    %467 = tensor.empty() : tensor<1x169x768xf32>
    %468 = hip.layer_norm(%arg0) ins(%466, %169, %168 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%467 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_158 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_159 = tensor.reshape %468(%cst_158) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %469 = tensor.empty() : tensor<169x3072xf32>
    %470 = hip.gemm(%arg0) ins(%reshape_159, %167, %166 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%469 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_160 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_161 = tensor.reshape %470(%cst_160) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %471 = tensor.empty() : tensor<1x169x3072xf32>
    %472 = hip.gelu(%arg0) ins(%reshape_161 : tensor<1x169x3072xf32>) outs(%471 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_162 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_163 = tensor.reshape %472(%cst_162) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %473 = tensor.empty() : tensor<169x768xf32>
    %474 = hip.gemm(%arg0) ins(%reshape_163, %165, %164 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%473 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_164 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_165 = tensor.reshape %474(%cst_164) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %475 = tensor.empty() : tensor<1x169x768xf32>
    %476 = hip.add(%arg0) ins(%468, %reshape_165 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%475 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_166 = arith.constant dense<768> : tensor<1xi64>
    %477 = tensor.empty() : tensor<1x169x768xf32>
    %478 = hip.layer_norm(%arg0) ins(%476, %163, %162 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%477 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_167 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_168 = tensor.reshape %478(%cst_167) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %479 = tensor.empty() : tensor<169x768xf32>
    %480 = hip.gemm(%arg0) ins(%reshape_168, %161, %160 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%479 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_169 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_170 = tensor.reshape %480(%cst_169) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_171 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_172 = tensor.reshape %reshape_170(%cst_171) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %481 = tensor.empty() : tensor<1x12x169x64xf32>
    %482 = hip.transpose(%arg0) ins(%reshape_172 : tensor<1x169x12x64xf32>) outs(%481 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_173 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_174 = tensor.reshape %478(%cst_173) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %483 = tensor.empty() : tensor<169x768xf32>
    %484 = hip.gemm(%arg0) ins(%reshape_174, %159, %158 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%483 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_175 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_176 = tensor.reshape %484(%cst_175) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_177 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_178 = tensor.reshape %reshape_176(%cst_177) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %485 = tensor.empty() : tensor<1x12x169x64xf32>
    %486 = hip.transpose(%arg0) ins(%reshape_178 : tensor<1x169x12x64xf32>) outs(%485 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_179 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_180 = tensor.reshape %478(%cst_179) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %487 = tensor.empty() : tensor<169x768xf32>
    %488 = hip.gemm(%arg0) ins(%reshape_180, %157, %156 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%487 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_181 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_182 = tensor.reshape %488(%cst_181) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_183 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_184 = tensor.reshape %reshape_182(%cst_183) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %489 = tensor.empty() : tensor<1x12x169x64xf32>
    %490 = hip.transpose(%arg0) ins(%reshape_184 : tensor<1x169x12x64xf32>) outs(%489 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_185 = arith.constant 0.000000e+00 : f32
    %cst_186 = arith.constant 0xFF800000 : f32
    %491 = hip.constant {serialization_order = 249 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_187 = tensor.splat %cst_185 : tensor<1x1x169x169xf32>
    %492 = hip.constant {serialization_order = 250 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_188 = tensor.splat %cst_186 : tensor<1x1x169x169xf32>
    %493 = tensor.empty() : tensor<1x1x169x169xf32>
    %494 = hip.where(%arg0) ins(%319, %491, %492 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%493 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %495 = tensor.empty() : tensor<1x169x12x64xf32>
    %496 = hip.transpose(%arg0) ins(%482 : tensor<1x12x169x64xf32>) outs(%495 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_189 = tensor.collapse_shape %496 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %497 = tensor.empty() : tensor<1x169x12x64xf32>
    %498 = hip.transpose(%arg0) ins(%486 : tensor<1x12x169x64xf32>) outs(%497 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_190 = tensor.collapse_shape %498 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %499 = tensor.empty() : tensor<1x169x12x64xf32>
    %500 = hip.transpose(%arg0) ins(%490 : tensor<1x12x169x64xf32>) outs(%499 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_191 = tensor.collapse_shape %500 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %501 = hip.constant {serialization_order = 251 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_192 = arith.constant dense<168> : tensor<1xi32>
    %502 = hip.constant {serialization_order = 252 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_193 = arith.constant dense<169> : tensor<i32>
    %503 = tensor.empty() : tensor<1x169x768xf32>
    %504 = tensor.empty() : tensor<1x12x169x64xf32>
    %505 = tensor.empty() : tensor<1x12x169x64xf32>
    %506:3 = hip.gqa(%arg0) ins(%collapsed_189, %collapsed_190, %collapsed_191, %501, %502, %494 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%503, %504, %505 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_194 = tensor.expand_shape %506#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %507 = tensor.empty() : tensor<1x12x169x64xf32>
    %508 = hip.transpose(%arg0) ins(%expanded_194 : tensor<1x169x12x64xf32>) outs(%507 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %509 = tensor.empty() : tensor<1x169x12x64xf32>
    %510 = hip.transpose(%arg0) ins(%508 : tensor<1x12x169x64xf32>) outs(%509 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_195 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_196 = tensor.reshape %510(%cst_195) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_197 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_198 = tensor.reshape %reshape_196(%cst_197) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %511 = tensor.empty() : tensor<169x768xf32>
    %512 = hip.gemm(%arg0) ins(%reshape_198, %155, %154 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%511 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_199 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_200 = tensor.reshape %512(%cst_199) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %513 = tensor.empty() : tensor<1x169x768xf32>
    %514 = hip.add(%arg0) ins(%478, %reshape_200 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%513 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_201 = arith.constant dense<768> : tensor<1xi64>
    %515 = tensor.empty() : tensor<1x169x768xf32>
    %516 = hip.layer_norm(%arg0) ins(%514, %153, %152 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%515 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_202 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_203 = tensor.reshape %516(%cst_202) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %517 = tensor.empty() : tensor<169x3072xf32>
    %518 = hip.gemm(%arg0) ins(%reshape_203, %151, %150 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%517 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_204 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_205 = tensor.reshape %518(%cst_204) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %519 = tensor.empty() : tensor<1x169x3072xf32>
    %520 = hip.gelu(%arg0) ins(%reshape_205 : tensor<1x169x3072xf32>) outs(%519 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_206 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_207 = tensor.reshape %520(%cst_206) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %521 = tensor.empty() : tensor<169x768xf32>
    %522 = hip.gemm(%arg0) ins(%reshape_207, %149, %148 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%521 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_208 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_209 = tensor.reshape %522(%cst_208) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %523 = tensor.empty() : tensor<1x169x768xf32>
    %524 = hip.add(%arg0) ins(%516, %reshape_209 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%523 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_210 = arith.constant dense<768> : tensor<1xi64>
    %525 = tensor.empty() : tensor<1x169x768xf32>
    %526 = hip.layer_norm(%arg0) ins(%524, %147, %146 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%525 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_211 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_212 = tensor.reshape %526(%cst_211) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %527 = tensor.empty() : tensor<169x768xf32>
    %528 = hip.gemm(%arg0) ins(%reshape_212, %145, %144 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%527 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_213 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_214 = tensor.reshape %528(%cst_213) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_215 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_216 = tensor.reshape %reshape_214(%cst_215) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %529 = tensor.empty() : tensor<1x12x169x64xf32>
    %530 = hip.transpose(%arg0) ins(%reshape_216 : tensor<1x169x12x64xf32>) outs(%529 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_217 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_218 = tensor.reshape %526(%cst_217) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %531 = tensor.empty() : tensor<169x768xf32>
    %532 = hip.gemm(%arg0) ins(%reshape_218, %143, %142 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%531 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_219 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_220 = tensor.reshape %532(%cst_219) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_221 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_222 = tensor.reshape %reshape_220(%cst_221) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %533 = tensor.empty() : tensor<1x12x169x64xf32>
    %534 = hip.transpose(%arg0) ins(%reshape_222 : tensor<1x169x12x64xf32>) outs(%533 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_223 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_224 = tensor.reshape %526(%cst_223) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %535 = tensor.empty() : tensor<169x768xf32>
    %536 = hip.gemm(%arg0) ins(%reshape_224, %141, %140 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%535 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_225 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_226 = tensor.reshape %536(%cst_225) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_227 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_228 = tensor.reshape %reshape_226(%cst_227) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %537 = tensor.empty() : tensor<1x12x169x64xf32>
    %538 = hip.transpose(%arg0) ins(%reshape_228 : tensor<1x169x12x64xf32>) outs(%537 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_229 = arith.constant 0.000000e+00 : f32
    %cst_230 = arith.constant 0xFF800000 : f32
    %539 = hip.constant {serialization_order = 253 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_231 = tensor.splat %cst_229 : tensor<1x1x169x169xf32>
    %540 = hip.constant {serialization_order = 254 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_232 = tensor.splat %cst_230 : tensor<1x1x169x169xf32>
    %541 = tensor.empty() : tensor<1x1x169x169xf32>
    %542 = hip.where(%arg0) ins(%319, %539, %540 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%541 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %543 = tensor.empty() : tensor<1x169x12x64xf32>
    %544 = hip.transpose(%arg0) ins(%530 : tensor<1x12x169x64xf32>) outs(%543 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_233 = tensor.collapse_shape %544 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %545 = tensor.empty() : tensor<1x169x12x64xf32>
    %546 = hip.transpose(%arg0) ins(%534 : tensor<1x12x169x64xf32>) outs(%545 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_234 = tensor.collapse_shape %546 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %547 = tensor.empty() : tensor<1x169x12x64xf32>
    %548 = hip.transpose(%arg0) ins(%538 : tensor<1x12x169x64xf32>) outs(%547 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_235 = tensor.collapse_shape %548 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %549 = hip.constant {serialization_order = 255 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_236 = arith.constant dense<168> : tensor<1xi32>
    %550 = hip.constant {serialization_order = 256 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_237 = arith.constant dense<169> : tensor<i32>
    %551 = tensor.empty() : tensor<1x169x768xf32>
    %552 = tensor.empty() : tensor<1x12x169x64xf32>
    %553 = tensor.empty() : tensor<1x12x169x64xf32>
    %554:3 = hip.gqa(%arg0) ins(%collapsed_233, %collapsed_234, %collapsed_235, %549, %550, %542 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%551, %552, %553 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_238 = tensor.expand_shape %554#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %555 = tensor.empty() : tensor<1x12x169x64xf32>
    %556 = hip.transpose(%arg0) ins(%expanded_238 : tensor<1x169x12x64xf32>) outs(%555 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %557 = tensor.empty() : tensor<1x169x12x64xf32>
    %558 = hip.transpose(%arg0) ins(%556 : tensor<1x12x169x64xf32>) outs(%557 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_239 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_240 = tensor.reshape %558(%cst_239) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_241 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_242 = tensor.reshape %reshape_240(%cst_241) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %559 = tensor.empty() : tensor<169x768xf32>
    %560 = hip.gemm(%arg0) ins(%reshape_242, %139, %138 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%559 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_243 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_244 = tensor.reshape %560(%cst_243) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %561 = tensor.empty() : tensor<1x169x768xf32>
    %562 = hip.add(%arg0) ins(%526, %reshape_244 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%561 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_245 = arith.constant dense<768> : tensor<1xi64>
    %563 = tensor.empty() : tensor<1x169x768xf32>
    %564 = hip.layer_norm(%arg0) ins(%562, %137, %136 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%563 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_246 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_247 = tensor.reshape %564(%cst_246) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %565 = tensor.empty() : tensor<169x3072xf32>
    %566 = hip.gemm(%arg0) ins(%reshape_247, %135, %134 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%565 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_248 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_249 = tensor.reshape %566(%cst_248) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %567 = tensor.empty() : tensor<1x169x3072xf32>
    %568 = hip.gelu(%arg0) ins(%reshape_249 : tensor<1x169x3072xf32>) outs(%567 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_250 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_251 = tensor.reshape %568(%cst_250) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %569 = tensor.empty() : tensor<169x768xf32>
    %570 = hip.gemm(%arg0) ins(%reshape_251, %133, %132 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%569 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_252 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_253 = tensor.reshape %570(%cst_252) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %571 = tensor.empty() : tensor<1x169x768xf32>
    %572 = hip.add(%arg0) ins(%564, %reshape_253 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%571 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_254 = arith.constant dense<768> : tensor<1xi64>
    %573 = tensor.empty() : tensor<1x169x768xf32>
    %574 = hip.layer_norm(%arg0) ins(%572, %131, %130 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%573 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_255 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_256 = tensor.reshape %574(%cst_255) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %575 = tensor.empty() : tensor<169x768xf32>
    %576 = hip.gemm(%arg0) ins(%reshape_256, %129, %128 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%575 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_257 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_258 = tensor.reshape %576(%cst_257) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_259 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_260 = tensor.reshape %reshape_258(%cst_259) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %577 = tensor.empty() : tensor<1x12x169x64xf32>
    %578 = hip.transpose(%arg0) ins(%reshape_260 : tensor<1x169x12x64xf32>) outs(%577 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_261 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_262 = tensor.reshape %574(%cst_261) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %579 = tensor.empty() : tensor<169x768xf32>
    %580 = hip.gemm(%arg0) ins(%reshape_262, %127, %126 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%579 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_263 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_264 = tensor.reshape %580(%cst_263) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_265 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_266 = tensor.reshape %reshape_264(%cst_265) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %581 = tensor.empty() : tensor<1x12x169x64xf32>
    %582 = hip.transpose(%arg0) ins(%reshape_266 : tensor<1x169x12x64xf32>) outs(%581 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_267 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_268 = tensor.reshape %574(%cst_267) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %583 = tensor.empty() : tensor<169x768xf32>
    %584 = hip.gemm(%arg0) ins(%reshape_268, %125, %124 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%583 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_269 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_270 = tensor.reshape %584(%cst_269) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_271 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_272 = tensor.reshape %reshape_270(%cst_271) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %585 = tensor.empty() : tensor<1x12x169x64xf32>
    %586 = hip.transpose(%arg0) ins(%reshape_272 : tensor<1x169x12x64xf32>) outs(%585 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_273 = arith.constant 0.000000e+00 : f32
    %cst_274 = arith.constant 0xFF800000 : f32
    %587 = hip.constant {serialization_order = 257 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_275 = tensor.splat %cst_273 : tensor<1x1x169x169xf32>
    %588 = hip.constant {serialization_order = 258 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_276 = tensor.splat %cst_274 : tensor<1x1x169x169xf32>
    %589 = tensor.empty() : tensor<1x1x169x169xf32>
    %590 = hip.where(%arg0) ins(%319, %587, %588 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%589 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %591 = tensor.empty() : tensor<1x169x12x64xf32>
    %592 = hip.transpose(%arg0) ins(%578 : tensor<1x12x169x64xf32>) outs(%591 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_277 = tensor.collapse_shape %592 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %593 = tensor.empty() : tensor<1x169x12x64xf32>
    %594 = hip.transpose(%arg0) ins(%582 : tensor<1x12x169x64xf32>) outs(%593 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_278 = tensor.collapse_shape %594 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %595 = tensor.empty() : tensor<1x169x12x64xf32>
    %596 = hip.transpose(%arg0) ins(%586 : tensor<1x12x169x64xf32>) outs(%595 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_279 = tensor.collapse_shape %596 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %597 = hip.constant {serialization_order = 259 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_280 = arith.constant dense<168> : tensor<1xi32>
    %598 = hip.constant {serialization_order = 260 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_281 = arith.constant dense<169> : tensor<i32>
    %599 = tensor.empty() : tensor<1x169x768xf32>
    %600 = tensor.empty() : tensor<1x12x169x64xf32>
    %601 = tensor.empty() : tensor<1x12x169x64xf32>
    %602:3 = hip.gqa(%arg0) ins(%collapsed_277, %collapsed_278, %collapsed_279, %597, %598, %590 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%599, %600, %601 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_282 = tensor.expand_shape %602#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %603 = tensor.empty() : tensor<1x12x169x64xf32>
    %604 = hip.transpose(%arg0) ins(%expanded_282 : tensor<1x169x12x64xf32>) outs(%603 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %605 = tensor.empty() : tensor<1x169x12x64xf32>
    %606 = hip.transpose(%arg0) ins(%604 : tensor<1x12x169x64xf32>) outs(%605 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_283 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_284 = tensor.reshape %606(%cst_283) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_285 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_286 = tensor.reshape %reshape_284(%cst_285) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %607 = tensor.empty() : tensor<169x768xf32>
    %608 = hip.gemm(%arg0) ins(%reshape_286, %123, %122 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%607 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_287 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_288 = tensor.reshape %608(%cst_287) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %609 = tensor.empty() : tensor<1x169x768xf32>
    %610 = hip.add(%arg0) ins(%574, %reshape_288 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%609 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_289 = arith.constant dense<768> : tensor<1xi64>
    %611 = tensor.empty() : tensor<1x169x768xf32>
    %612 = hip.layer_norm(%arg0) ins(%610, %121, %120 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%611 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_290 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_291 = tensor.reshape %612(%cst_290) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %613 = tensor.empty() : tensor<169x3072xf32>
    %614 = hip.gemm(%arg0) ins(%reshape_291, %119, %118 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%613 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_292 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_293 = tensor.reshape %614(%cst_292) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %615 = tensor.empty() : tensor<1x169x3072xf32>
    %616 = hip.gelu(%arg0) ins(%reshape_293 : tensor<1x169x3072xf32>) outs(%615 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_294 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_295 = tensor.reshape %616(%cst_294) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %617 = tensor.empty() : tensor<169x768xf32>
    %618 = hip.gemm(%arg0) ins(%reshape_295, %117, %116 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%617 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_296 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_297 = tensor.reshape %618(%cst_296) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %619 = tensor.empty() : tensor<1x169x768xf32>
    %620 = hip.add(%arg0) ins(%612, %reshape_297 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%619 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_298 = arith.constant dense<768> : tensor<1xi64>
    %621 = tensor.empty() : tensor<1x169x768xf32>
    %622 = hip.layer_norm(%arg0) ins(%620, %115, %114 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%621 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_299 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_300 = tensor.reshape %622(%cst_299) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %623 = tensor.empty() : tensor<169x768xf32>
    %624 = hip.gemm(%arg0) ins(%reshape_300, %113, %112 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%623 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_301 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_302 = tensor.reshape %624(%cst_301) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_303 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_304 = tensor.reshape %reshape_302(%cst_303) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %625 = tensor.empty() : tensor<1x12x169x64xf32>
    %626 = hip.transpose(%arg0) ins(%reshape_304 : tensor<1x169x12x64xf32>) outs(%625 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_305 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_306 = tensor.reshape %622(%cst_305) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %627 = tensor.empty() : tensor<169x768xf32>
    %628 = hip.gemm(%arg0) ins(%reshape_306, %111, %110 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%627 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_307 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_308 = tensor.reshape %628(%cst_307) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_309 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_310 = tensor.reshape %reshape_308(%cst_309) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %629 = tensor.empty() : tensor<1x12x169x64xf32>
    %630 = hip.transpose(%arg0) ins(%reshape_310 : tensor<1x169x12x64xf32>) outs(%629 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_311 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_312 = tensor.reshape %622(%cst_311) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %631 = tensor.empty() : tensor<169x768xf32>
    %632 = hip.gemm(%arg0) ins(%reshape_312, %109, %108 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%631 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_313 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_314 = tensor.reshape %632(%cst_313) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_315 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_316 = tensor.reshape %reshape_314(%cst_315) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %633 = tensor.empty() : tensor<1x12x169x64xf32>
    %634 = hip.transpose(%arg0) ins(%reshape_316 : tensor<1x169x12x64xf32>) outs(%633 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_317 = arith.constant 0.000000e+00 : f32
    %cst_318 = arith.constant 0xFF800000 : f32
    %635 = hip.constant {serialization_order = 261 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_319 = tensor.splat %cst_317 : tensor<1x1x169x169xf32>
    %636 = hip.constant {serialization_order = 262 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_320 = tensor.splat %cst_318 : tensor<1x1x169x169xf32>
    %637 = tensor.empty() : tensor<1x1x169x169xf32>
    %638 = hip.where(%arg0) ins(%319, %635, %636 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%637 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %639 = tensor.empty() : tensor<1x169x12x64xf32>
    %640 = hip.transpose(%arg0) ins(%626 : tensor<1x12x169x64xf32>) outs(%639 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_321 = tensor.collapse_shape %640 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %641 = tensor.empty() : tensor<1x169x12x64xf32>
    %642 = hip.transpose(%arg0) ins(%630 : tensor<1x12x169x64xf32>) outs(%641 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_322 = tensor.collapse_shape %642 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %643 = tensor.empty() : tensor<1x169x12x64xf32>
    %644 = hip.transpose(%arg0) ins(%634 : tensor<1x12x169x64xf32>) outs(%643 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_323 = tensor.collapse_shape %644 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %645 = hip.constant {serialization_order = 263 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_324 = arith.constant dense<168> : tensor<1xi32>
    %646 = hip.constant {serialization_order = 264 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_325 = arith.constant dense<169> : tensor<i32>
    %647 = tensor.empty() : tensor<1x169x768xf32>
    %648 = tensor.empty() : tensor<1x12x169x64xf32>
    %649 = tensor.empty() : tensor<1x12x169x64xf32>
    %650:3 = hip.gqa(%arg0) ins(%collapsed_321, %collapsed_322, %collapsed_323, %645, %646, %638 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%647, %648, %649 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_326 = tensor.expand_shape %650#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %651 = tensor.empty() : tensor<1x12x169x64xf32>
    %652 = hip.transpose(%arg0) ins(%expanded_326 : tensor<1x169x12x64xf32>) outs(%651 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %653 = tensor.empty() : tensor<1x169x12x64xf32>
    %654 = hip.transpose(%arg0) ins(%652 : tensor<1x12x169x64xf32>) outs(%653 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_327 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_328 = tensor.reshape %654(%cst_327) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_329 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_330 = tensor.reshape %reshape_328(%cst_329) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %655 = tensor.empty() : tensor<169x768xf32>
    %656 = hip.gemm(%arg0) ins(%reshape_330, %107, %106 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%655 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_331 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_332 = tensor.reshape %656(%cst_331) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %657 = tensor.empty() : tensor<1x169x768xf32>
    %658 = hip.add(%arg0) ins(%622, %reshape_332 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%657 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_333 = arith.constant dense<768> : tensor<1xi64>
    %659 = tensor.empty() : tensor<1x169x768xf32>
    %660 = hip.layer_norm(%arg0) ins(%658, %105, %104 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%659 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_334 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_335 = tensor.reshape %660(%cst_334) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %661 = tensor.empty() : tensor<169x3072xf32>
    %662 = hip.gemm(%arg0) ins(%reshape_335, %103, %102 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%661 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_336 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_337 = tensor.reshape %662(%cst_336) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %663 = tensor.empty() : tensor<1x169x3072xf32>
    %664 = hip.gelu(%arg0) ins(%reshape_337 : tensor<1x169x3072xf32>) outs(%663 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_338 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_339 = tensor.reshape %664(%cst_338) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %665 = tensor.empty() : tensor<169x768xf32>
    %666 = hip.gemm(%arg0) ins(%reshape_339, %101, %100 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%665 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_340 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_341 = tensor.reshape %666(%cst_340) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %667 = tensor.empty() : tensor<1x169x768xf32>
    %668 = hip.add(%arg0) ins(%660, %reshape_341 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%667 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_342 = arith.constant dense<768> : tensor<1xi64>
    %669 = tensor.empty() : tensor<1x169x768xf32>
    %670 = hip.layer_norm(%arg0) ins(%668, %99, %98 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%669 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_343 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_344 = tensor.reshape %670(%cst_343) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %671 = tensor.empty() : tensor<169x768xf32>
    %672 = hip.gemm(%arg0) ins(%reshape_344, %97, %96 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%671 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_345 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_346 = tensor.reshape %672(%cst_345) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_347 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_348 = tensor.reshape %reshape_346(%cst_347) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %673 = tensor.empty() : tensor<1x12x169x64xf32>
    %674 = hip.transpose(%arg0) ins(%reshape_348 : tensor<1x169x12x64xf32>) outs(%673 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_349 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_350 = tensor.reshape %670(%cst_349) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %675 = tensor.empty() : tensor<169x768xf32>
    %676 = hip.gemm(%arg0) ins(%reshape_350, %95, %94 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%675 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_351 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_352 = tensor.reshape %676(%cst_351) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_353 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_354 = tensor.reshape %reshape_352(%cst_353) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %677 = tensor.empty() : tensor<1x12x169x64xf32>
    %678 = hip.transpose(%arg0) ins(%reshape_354 : tensor<1x169x12x64xf32>) outs(%677 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_355 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_356 = tensor.reshape %670(%cst_355) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %679 = tensor.empty() : tensor<169x768xf32>
    %680 = hip.gemm(%arg0) ins(%reshape_356, %93, %92 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%679 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_357 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_358 = tensor.reshape %680(%cst_357) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_359 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_360 = tensor.reshape %reshape_358(%cst_359) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %681 = tensor.empty() : tensor<1x12x169x64xf32>
    %682 = hip.transpose(%arg0) ins(%reshape_360 : tensor<1x169x12x64xf32>) outs(%681 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_361 = arith.constant 0.000000e+00 : f32
    %cst_362 = arith.constant 0xFF800000 : f32
    %683 = hip.constant {serialization_order = 265 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_363 = tensor.splat %cst_361 : tensor<1x1x169x169xf32>
    %684 = hip.constant {serialization_order = 266 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_364 = tensor.splat %cst_362 : tensor<1x1x169x169xf32>
    %685 = tensor.empty() : tensor<1x1x169x169xf32>
    %686 = hip.where(%arg0) ins(%319, %683, %684 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%685 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %687 = tensor.empty() : tensor<1x169x12x64xf32>
    %688 = hip.transpose(%arg0) ins(%674 : tensor<1x12x169x64xf32>) outs(%687 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_365 = tensor.collapse_shape %688 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %689 = tensor.empty() : tensor<1x169x12x64xf32>
    %690 = hip.transpose(%arg0) ins(%678 : tensor<1x12x169x64xf32>) outs(%689 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_366 = tensor.collapse_shape %690 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %691 = tensor.empty() : tensor<1x169x12x64xf32>
    %692 = hip.transpose(%arg0) ins(%682 : tensor<1x12x169x64xf32>) outs(%691 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_367 = tensor.collapse_shape %692 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %693 = hip.constant {serialization_order = 267 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_368 = arith.constant dense<168> : tensor<1xi32>
    %694 = hip.constant {serialization_order = 268 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_369 = arith.constant dense<169> : tensor<i32>
    %695 = tensor.empty() : tensor<1x169x768xf32>
    %696 = tensor.empty() : tensor<1x12x169x64xf32>
    %697 = tensor.empty() : tensor<1x12x169x64xf32>
    %698:3 = hip.gqa(%arg0) ins(%collapsed_365, %collapsed_366, %collapsed_367, %693, %694, %686 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%695, %696, %697 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_370 = tensor.expand_shape %698#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %699 = tensor.empty() : tensor<1x12x169x64xf32>
    %700 = hip.transpose(%arg0) ins(%expanded_370 : tensor<1x169x12x64xf32>) outs(%699 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %701 = tensor.empty() : tensor<1x169x12x64xf32>
    %702 = hip.transpose(%arg0) ins(%700 : tensor<1x12x169x64xf32>) outs(%701 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_371 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_372 = tensor.reshape %702(%cst_371) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_373 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_374 = tensor.reshape %reshape_372(%cst_373) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %703 = tensor.empty() : tensor<169x768xf32>
    %704 = hip.gemm(%arg0) ins(%reshape_374, %91, %90 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%703 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_375 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_376 = tensor.reshape %704(%cst_375) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %705 = tensor.empty() : tensor<1x169x768xf32>
    %706 = hip.add(%arg0) ins(%670, %reshape_376 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%705 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_377 = arith.constant dense<768> : tensor<1xi64>
    %707 = tensor.empty() : tensor<1x169x768xf32>
    %708 = hip.layer_norm(%arg0) ins(%706, %89, %88 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%707 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_378 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_379 = tensor.reshape %708(%cst_378) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %709 = tensor.empty() : tensor<169x3072xf32>
    %710 = hip.gemm(%arg0) ins(%reshape_379, %87, %86 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%709 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_380 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_381 = tensor.reshape %710(%cst_380) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %711 = tensor.empty() : tensor<1x169x3072xf32>
    %712 = hip.gelu(%arg0) ins(%reshape_381 : tensor<1x169x3072xf32>) outs(%711 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_382 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_383 = tensor.reshape %712(%cst_382) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %713 = tensor.empty() : tensor<169x768xf32>
    %714 = hip.gemm(%arg0) ins(%reshape_383, %85, %84 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%713 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_384 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_385 = tensor.reshape %714(%cst_384) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %715 = tensor.empty() : tensor<1x169x768xf32>
    %716 = hip.add(%arg0) ins(%708, %reshape_385 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%715 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_386 = arith.constant dense<768> : tensor<1xi64>
    %717 = tensor.empty() : tensor<1x169x768xf32>
    %718 = hip.layer_norm(%arg0) ins(%716, %83, %82 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%717 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_387 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_388 = tensor.reshape %718(%cst_387) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %719 = tensor.empty() : tensor<169x768xf32>
    %720 = hip.gemm(%arg0) ins(%reshape_388, %81, %80 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%719 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_389 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_390 = tensor.reshape %720(%cst_389) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_391 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_392 = tensor.reshape %reshape_390(%cst_391) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %721 = tensor.empty() : tensor<1x12x169x64xf32>
    %722 = hip.transpose(%arg0) ins(%reshape_392 : tensor<1x169x12x64xf32>) outs(%721 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_393 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_394 = tensor.reshape %718(%cst_393) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %723 = tensor.empty() : tensor<169x768xf32>
    %724 = hip.gemm(%arg0) ins(%reshape_394, %79, %78 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%723 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_395 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_396 = tensor.reshape %724(%cst_395) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_397 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_398 = tensor.reshape %reshape_396(%cst_397) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %725 = tensor.empty() : tensor<1x12x169x64xf32>
    %726 = hip.transpose(%arg0) ins(%reshape_398 : tensor<1x169x12x64xf32>) outs(%725 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_399 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_400 = tensor.reshape %718(%cst_399) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %727 = tensor.empty() : tensor<169x768xf32>
    %728 = hip.gemm(%arg0) ins(%reshape_400, %77, %76 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%727 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_401 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_402 = tensor.reshape %728(%cst_401) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_403 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_404 = tensor.reshape %reshape_402(%cst_403) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %729 = tensor.empty() : tensor<1x12x169x64xf32>
    %730 = hip.transpose(%arg0) ins(%reshape_404 : tensor<1x169x12x64xf32>) outs(%729 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_405 = arith.constant 0.000000e+00 : f32
    %cst_406 = arith.constant 0xFF800000 : f32
    %731 = hip.constant {serialization_order = 269 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_407 = tensor.splat %cst_405 : tensor<1x1x169x169xf32>
    %732 = hip.constant {serialization_order = 270 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_408 = tensor.splat %cst_406 : tensor<1x1x169x169xf32>
    %733 = tensor.empty() : tensor<1x1x169x169xf32>
    %734 = hip.where(%arg0) ins(%319, %731, %732 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%733 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %735 = tensor.empty() : tensor<1x169x12x64xf32>
    %736 = hip.transpose(%arg0) ins(%722 : tensor<1x12x169x64xf32>) outs(%735 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_409 = tensor.collapse_shape %736 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %737 = tensor.empty() : tensor<1x169x12x64xf32>
    %738 = hip.transpose(%arg0) ins(%726 : tensor<1x12x169x64xf32>) outs(%737 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_410 = tensor.collapse_shape %738 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %739 = tensor.empty() : tensor<1x169x12x64xf32>
    %740 = hip.transpose(%arg0) ins(%730 : tensor<1x12x169x64xf32>) outs(%739 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_411 = tensor.collapse_shape %740 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %741 = hip.constant {serialization_order = 271 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_412 = arith.constant dense<168> : tensor<1xi32>
    %742 = hip.constant {serialization_order = 272 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_413 = arith.constant dense<169> : tensor<i32>
    %743 = tensor.empty() : tensor<1x169x768xf32>
    %744 = tensor.empty() : tensor<1x12x169x64xf32>
    %745 = tensor.empty() : tensor<1x12x169x64xf32>
    %746:3 = hip.gqa(%arg0) ins(%collapsed_409, %collapsed_410, %collapsed_411, %741, %742, %734 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%743, %744, %745 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_414 = tensor.expand_shape %746#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %747 = tensor.empty() : tensor<1x12x169x64xf32>
    %748 = hip.transpose(%arg0) ins(%expanded_414 : tensor<1x169x12x64xf32>) outs(%747 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %749 = tensor.empty() : tensor<1x169x12x64xf32>
    %750 = hip.transpose(%arg0) ins(%748 : tensor<1x12x169x64xf32>) outs(%749 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_415 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_416 = tensor.reshape %750(%cst_415) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_417 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_418 = tensor.reshape %reshape_416(%cst_417) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %751 = tensor.empty() : tensor<169x768xf32>
    %752 = hip.gemm(%arg0) ins(%reshape_418, %75, %74 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%751 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_419 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_420 = tensor.reshape %752(%cst_419) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %753 = tensor.empty() : tensor<1x169x768xf32>
    %754 = hip.add(%arg0) ins(%718, %reshape_420 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%753 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_421 = arith.constant dense<768> : tensor<1xi64>
    %755 = tensor.empty() : tensor<1x169x768xf32>
    %756 = hip.layer_norm(%arg0) ins(%754, %73, %72 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%755 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_422 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_423 = tensor.reshape %756(%cst_422) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %757 = tensor.empty() : tensor<169x3072xf32>
    %758 = hip.gemm(%arg0) ins(%reshape_423, %71, %70 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%757 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_424 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_425 = tensor.reshape %758(%cst_424) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %759 = tensor.empty() : tensor<1x169x3072xf32>
    %760 = hip.gelu(%arg0) ins(%reshape_425 : tensor<1x169x3072xf32>) outs(%759 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_426 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_427 = tensor.reshape %760(%cst_426) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %761 = tensor.empty() : tensor<169x768xf32>
    %762 = hip.gemm(%arg0) ins(%reshape_427, %69, %68 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%761 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_428 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_429 = tensor.reshape %762(%cst_428) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %763 = tensor.empty() : tensor<1x169x768xf32>
    %764 = hip.add(%arg0) ins(%756, %reshape_429 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%763 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_430 = arith.constant dense<768> : tensor<1xi64>
    %765 = tensor.empty() : tensor<1x169x768xf32>
    %766 = hip.layer_norm(%arg0) ins(%764, %67, %66 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%765 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_431 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_432 = tensor.reshape %766(%cst_431) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %767 = tensor.empty() : tensor<169x768xf32>
    %768 = hip.gemm(%arg0) ins(%reshape_432, %65, %64 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%767 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_433 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_434 = tensor.reshape %768(%cst_433) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_435 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_436 = tensor.reshape %reshape_434(%cst_435) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %769 = tensor.empty() : tensor<1x12x169x64xf32>
    %770 = hip.transpose(%arg0) ins(%reshape_436 : tensor<1x169x12x64xf32>) outs(%769 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_437 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_438 = tensor.reshape %766(%cst_437) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %771 = tensor.empty() : tensor<169x768xf32>
    %772 = hip.gemm(%arg0) ins(%reshape_438, %63, %62 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%771 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_439 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_440 = tensor.reshape %772(%cst_439) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_441 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_442 = tensor.reshape %reshape_440(%cst_441) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %773 = tensor.empty() : tensor<1x12x169x64xf32>
    %774 = hip.transpose(%arg0) ins(%reshape_442 : tensor<1x169x12x64xf32>) outs(%773 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_443 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_444 = tensor.reshape %766(%cst_443) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %775 = tensor.empty() : tensor<169x768xf32>
    %776 = hip.gemm(%arg0) ins(%reshape_444, %61, %60 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%775 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_445 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_446 = tensor.reshape %776(%cst_445) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_447 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_448 = tensor.reshape %reshape_446(%cst_447) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %777 = tensor.empty() : tensor<1x12x169x64xf32>
    %778 = hip.transpose(%arg0) ins(%reshape_448 : tensor<1x169x12x64xf32>) outs(%777 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_449 = arith.constant 0.000000e+00 : f32
    %cst_450 = arith.constant 0xFF800000 : f32
    %779 = hip.constant {serialization_order = 273 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_451 = tensor.splat %cst_449 : tensor<1x1x169x169xf32>
    %780 = hip.constant {serialization_order = 274 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_452 = tensor.splat %cst_450 : tensor<1x1x169x169xf32>
    %781 = tensor.empty() : tensor<1x1x169x169xf32>
    %782 = hip.where(%arg0) ins(%319, %779, %780 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%781 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %783 = tensor.empty() : tensor<1x169x12x64xf32>
    %784 = hip.transpose(%arg0) ins(%770 : tensor<1x12x169x64xf32>) outs(%783 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_453 = tensor.collapse_shape %784 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %785 = tensor.empty() : tensor<1x169x12x64xf32>
    %786 = hip.transpose(%arg0) ins(%774 : tensor<1x12x169x64xf32>) outs(%785 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_454 = tensor.collapse_shape %786 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %787 = tensor.empty() : tensor<1x169x12x64xf32>
    %788 = hip.transpose(%arg0) ins(%778 : tensor<1x12x169x64xf32>) outs(%787 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_455 = tensor.collapse_shape %788 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %789 = hip.constant {serialization_order = 275 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_456 = arith.constant dense<168> : tensor<1xi32>
    %790 = hip.constant {serialization_order = 276 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_457 = arith.constant dense<169> : tensor<i32>
    %791 = tensor.empty() : tensor<1x169x768xf32>
    %792 = tensor.empty() : tensor<1x12x169x64xf32>
    %793 = tensor.empty() : tensor<1x12x169x64xf32>
    %794:3 = hip.gqa(%arg0) ins(%collapsed_453, %collapsed_454, %collapsed_455, %789, %790, %782 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%791, %792, %793 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_458 = tensor.expand_shape %794#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %795 = tensor.empty() : tensor<1x12x169x64xf32>
    %796 = hip.transpose(%arg0) ins(%expanded_458 : tensor<1x169x12x64xf32>) outs(%795 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %797 = tensor.empty() : tensor<1x169x12x64xf32>
    %798 = hip.transpose(%arg0) ins(%796 : tensor<1x12x169x64xf32>) outs(%797 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_459 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_460 = tensor.reshape %798(%cst_459) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_461 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_462 = tensor.reshape %reshape_460(%cst_461) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %799 = tensor.empty() : tensor<169x768xf32>
    %800 = hip.gemm(%arg0) ins(%reshape_462, %59, %58 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%799 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_463 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_464 = tensor.reshape %800(%cst_463) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %801 = tensor.empty() : tensor<1x169x768xf32>
    %802 = hip.add(%arg0) ins(%766, %reshape_464 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%801 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_465 = arith.constant dense<768> : tensor<1xi64>
    %803 = tensor.empty() : tensor<1x169x768xf32>
    %804 = hip.layer_norm(%arg0) ins(%802, %57, %56 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%803 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_466 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_467 = tensor.reshape %804(%cst_466) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %805 = tensor.empty() : tensor<169x3072xf32>
    %806 = hip.gemm(%arg0) ins(%reshape_467, %55, %54 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%805 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_468 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_469 = tensor.reshape %806(%cst_468) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %807 = tensor.empty() : tensor<1x169x3072xf32>
    %808 = hip.gelu(%arg0) ins(%reshape_469 : tensor<1x169x3072xf32>) outs(%807 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_470 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_471 = tensor.reshape %808(%cst_470) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %809 = tensor.empty() : tensor<169x768xf32>
    %810 = hip.gemm(%arg0) ins(%reshape_471, %53, %52 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%809 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_472 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_473 = tensor.reshape %810(%cst_472) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %811 = tensor.empty() : tensor<1x169x768xf32>
    %812 = hip.add(%arg0) ins(%804, %reshape_473 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%811 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_474 = arith.constant dense<768> : tensor<1xi64>
    %813 = tensor.empty() : tensor<1x169x768xf32>
    %814 = hip.layer_norm(%arg0) ins(%812, %51, %50 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%813 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_475 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_476 = tensor.reshape %814(%cst_475) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %815 = tensor.empty() : tensor<169x768xf32>
    %816 = hip.gemm(%arg0) ins(%reshape_476, %49, %48 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%815 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_477 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_478 = tensor.reshape %816(%cst_477) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_479 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_480 = tensor.reshape %reshape_478(%cst_479) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %817 = tensor.empty() : tensor<1x12x169x64xf32>
    %818 = hip.transpose(%arg0) ins(%reshape_480 : tensor<1x169x12x64xf32>) outs(%817 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_481 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_482 = tensor.reshape %814(%cst_481) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %819 = tensor.empty() : tensor<169x768xf32>
    %820 = hip.gemm(%arg0) ins(%reshape_482, %47, %46 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%819 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_483 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_484 = tensor.reshape %820(%cst_483) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_485 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_486 = tensor.reshape %reshape_484(%cst_485) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %821 = tensor.empty() : tensor<1x12x169x64xf32>
    %822 = hip.transpose(%arg0) ins(%reshape_486 : tensor<1x169x12x64xf32>) outs(%821 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_487 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_488 = tensor.reshape %814(%cst_487) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %823 = tensor.empty() : tensor<169x768xf32>
    %824 = hip.gemm(%arg0) ins(%reshape_488, %45, %44 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%823 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_489 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_490 = tensor.reshape %824(%cst_489) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_491 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_492 = tensor.reshape %reshape_490(%cst_491) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %825 = tensor.empty() : tensor<1x12x169x64xf32>
    %826 = hip.transpose(%arg0) ins(%reshape_492 : tensor<1x169x12x64xf32>) outs(%825 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_493 = arith.constant 0.000000e+00 : f32
    %cst_494 = arith.constant 0xFF800000 : f32
    %827 = hip.constant {serialization_order = 277 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_495 = tensor.splat %cst_493 : tensor<1x1x169x169xf32>
    %828 = hip.constant {serialization_order = 278 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_496 = tensor.splat %cst_494 : tensor<1x1x169x169xf32>
    %829 = tensor.empty() : tensor<1x1x169x169xf32>
    %830 = hip.where(%arg0) ins(%319, %827, %828 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%829 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %831 = tensor.empty() : tensor<1x169x12x64xf32>
    %832 = hip.transpose(%arg0) ins(%818 : tensor<1x12x169x64xf32>) outs(%831 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_497 = tensor.collapse_shape %832 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %833 = tensor.empty() : tensor<1x169x12x64xf32>
    %834 = hip.transpose(%arg0) ins(%822 : tensor<1x12x169x64xf32>) outs(%833 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_498 = tensor.collapse_shape %834 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %835 = tensor.empty() : tensor<1x169x12x64xf32>
    %836 = hip.transpose(%arg0) ins(%826 : tensor<1x12x169x64xf32>) outs(%835 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_499 = tensor.collapse_shape %836 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %837 = hip.constant {serialization_order = 279 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_500 = arith.constant dense<168> : tensor<1xi32>
    %838 = hip.constant {serialization_order = 280 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_501 = arith.constant dense<169> : tensor<i32>
    %839 = tensor.empty() : tensor<1x169x768xf32>
    %840 = tensor.empty() : tensor<1x12x169x64xf32>
    %841 = tensor.empty() : tensor<1x12x169x64xf32>
    %842:3 = hip.gqa(%arg0) ins(%collapsed_497, %collapsed_498, %collapsed_499, %837, %838, %830 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%839, %840, %841 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_502 = tensor.expand_shape %842#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %843 = tensor.empty() : tensor<1x12x169x64xf32>
    %844 = hip.transpose(%arg0) ins(%expanded_502 : tensor<1x169x12x64xf32>) outs(%843 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %845 = tensor.empty() : tensor<1x169x12x64xf32>
    %846 = hip.transpose(%arg0) ins(%844 : tensor<1x12x169x64xf32>) outs(%845 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_503 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_504 = tensor.reshape %846(%cst_503) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_505 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_506 = tensor.reshape %reshape_504(%cst_505) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %847 = tensor.empty() : tensor<169x768xf32>
    %848 = hip.gemm(%arg0) ins(%reshape_506, %43, %42 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%847 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_507 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_508 = tensor.reshape %848(%cst_507) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %849 = tensor.empty() : tensor<1x169x768xf32>
    %850 = hip.add(%arg0) ins(%814, %reshape_508 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%849 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_509 = arith.constant dense<768> : tensor<1xi64>
    %851 = tensor.empty() : tensor<1x169x768xf32>
    %852 = hip.layer_norm(%arg0) ins(%850, %41, %40 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%851 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_510 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_511 = tensor.reshape %852(%cst_510) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %853 = tensor.empty() : tensor<169x3072xf32>
    %854 = hip.gemm(%arg0) ins(%reshape_511, %39, %38 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%853 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_512 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_513 = tensor.reshape %854(%cst_512) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %855 = tensor.empty() : tensor<1x169x3072xf32>
    %856 = hip.gelu(%arg0) ins(%reshape_513 : tensor<1x169x3072xf32>) outs(%855 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_514 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_515 = tensor.reshape %856(%cst_514) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %857 = tensor.empty() : tensor<169x768xf32>
    %858 = hip.gemm(%arg0) ins(%reshape_515, %37, %36 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%857 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_516 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_517 = tensor.reshape %858(%cst_516) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %859 = tensor.empty() : tensor<1x169x768xf32>
    %860 = hip.add(%arg0) ins(%852, %reshape_517 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%859 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_518 = arith.constant dense<768> : tensor<1xi64>
    %861 = tensor.empty() : tensor<1x169x768xf32>
    %862 = hip.layer_norm(%arg0) ins(%860, %35, %34 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%861 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_519 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_520 = tensor.reshape %862(%cst_519) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %863 = tensor.empty() : tensor<169x768xf32>
    %864 = hip.gemm(%arg0) ins(%reshape_520, %33, %32 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%863 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_521 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_522 = tensor.reshape %864(%cst_521) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_523 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_524 = tensor.reshape %reshape_522(%cst_523) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %865 = tensor.empty() : tensor<1x12x169x64xf32>
    %866 = hip.transpose(%arg0) ins(%reshape_524 : tensor<1x169x12x64xf32>) outs(%865 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_525 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_526 = tensor.reshape %862(%cst_525) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %867 = tensor.empty() : tensor<169x768xf32>
    %868 = hip.gemm(%arg0) ins(%reshape_526, %31, %30 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%867 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_527 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_528 = tensor.reshape %868(%cst_527) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_529 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_530 = tensor.reshape %reshape_528(%cst_529) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %869 = tensor.empty() : tensor<1x12x169x64xf32>
    %870 = hip.transpose(%arg0) ins(%reshape_530 : tensor<1x169x12x64xf32>) outs(%869 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_531 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_532 = tensor.reshape %862(%cst_531) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %871 = tensor.empty() : tensor<169x768xf32>
    %872 = hip.gemm(%arg0) ins(%reshape_532, %29, %28 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%871 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_533 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_534 = tensor.reshape %872(%cst_533) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_535 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_536 = tensor.reshape %reshape_534(%cst_535) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %873 = tensor.empty() : tensor<1x12x169x64xf32>
    %874 = hip.transpose(%arg0) ins(%reshape_536 : tensor<1x169x12x64xf32>) outs(%873 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_537 = arith.constant 0.000000e+00 : f32
    %cst_538 = arith.constant 0xFF800000 : f32
    %875 = hip.constant {serialization_order = 281 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_539 = tensor.splat %cst_537 : tensor<1x1x169x169xf32>
    %876 = hip.constant {serialization_order = 282 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_540 = tensor.splat %cst_538 : tensor<1x1x169x169xf32>
    %877 = tensor.empty() : tensor<1x1x169x169xf32>
    %878 = hip.where(%arg0) ins(%319, %875, %876 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%877 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %879 = tensor.empty() : tensor<1x169x12x64xf32>
    %880 = hip.transpose(%arg0) ins(%866 : tensor<1x12x169x64xf32>) outs(%879 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_541 = tensor.collapse_shape %880 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %881 = tensor.empty() : tensor<1x169x12x64xf32>
    %882 = hip.transpose(%arg0) ins(%870 : tensor<1x12x169x64xf32>) outs(%881 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_542 = tensor.collapse_shape %882 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %883 = tensor.empty() : tensor<1x169x12x64xf32>
    %884 = hip.transpose(%arg0) ins(%874 : tensor<1x12x169x64xf32>) outs(%883 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_543 = tensor.collapse_shape %884 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %885 = hip.constant {serialization_order = 283 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_544 = arith.constant dense<168> : tensor<1xi32>
    %886 = hip.constant {serialization_order = 284 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_545 = arith.constant dense<169> : tensor<i32>
    %887 = tensor.empty() : tensor<1x169x768xf32>
    %888 = tensor.empty() : tensor<1x12x169x64xf32>
    %889 = tensor.empty() : tensor<1x12x169x64xf32>
    %890:3 = hip.gqa(%arg0) ins(%collapsed_541, %collapsed_542, %collapsed_543, %885, %886, %878 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%887, %888, %889 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_546 = tensor.expand_shape %890#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %891 = tensor.empty() : tensor<1x12x169x64xf32>
    %892 = hip.transpose(%arg0) ins(%expanded_546 : tensor<1x169x12x64xf32>) outs(%891 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %893 = tensor.empty() : tensor<1x169x12x64xf32>
    %894 = hip.transpose(%arg0) ins(%892 : tensor<1x12x169x64xf32>) outs(%893 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_547 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_548 = tensor.reshape %894(%cst_547) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_549 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_550 = tensor.reshape %reshape_548(%cst_549) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %895 = tensor.empty() : tensor<169x768xf32>
    %896 = hip.gemm(%arg0) ins(%reshape_550, %27, %26 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%895 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_551 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_552 = tensor.reshape %896(%cst_551) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %897 = tensor.empty() : tensor<1x169x768xf32>
    %898 = hip.add(%arg0) ins(%862, %reshape_552 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%897 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_553 = arith.constant dense<768> : tensor<1xi64>
    %899 = tensor.empty() : tensor<1x169x768xf32>
    %900 = hip.layer_norm(%arg0) ins(%898, %25, %24 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%899 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_554 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_555 = tensor.reshape %900(%cst_554) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %901 = tensor.empty() : tensor<169x3072xf32>
    %902 = hip.gemm(%arg0) ins(%reshape_555, %23, %22 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%901 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_556 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_557 = tensor.reshape %902(%cst_556) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %903 = tensor.empty() : tensor<1x169x3072xf32>
    %904 = hip.gelu(%arg0) ins(%reshape_557 : tensor<1x169x3072xf32>) outs(%903 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_558 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_559 = tensor.reshape %904(%cst_558) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %905 = tensor.empty() : tensor<169x768xf32>
    %906 = hip.gemm(%arg0) ins(%reshape_559, %21, %20 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%905 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_560 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_561 = tensor.reshape %906(%cst_560) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %907 = tensor.empty() : tensor<1x169x768xf32>
    %908 = hip.add(%arg0) ins(%900, %reshape_561 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%907 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_562 = arith.constant dense<768> : tensor<1xi64>
    %909 = tensor.empty() : tensor<1x169x768xf32>
    %910 = hip.layer_norm(%arg0) ins(%908, %19, %18 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%909 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_563 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_564 = tensor.reshape %910(%cst_563) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %911 = tensor.empty() : tensor<169x768xf32>
    %912 = hip.gemm(%arg0) ins(%reshape_564, %17, %16 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%911 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_565 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_566 = tensor.reshape %912(%cst_565) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_567 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_568 = tensor.reshape %reshape_566(%cst_567) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %913 = tensor.empty() : tensor<1x12x169x64xf32>
    %914 = hip.transpose(%arg0) ins(%reshape_568 : tensor<1x169x12x64xf32>) outs(%913 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_569 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_570 = tensor.reshape %910(%cst_569) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %915 = tensor.empty() : tensor<169x768xf32>
    %916 = hip.gemm(%arg0) ins(%reshape_570, %15, %14 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%915 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_571 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_572 = tensor.reshape %916(%cst_571) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_573 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_574 = tensor.reshape %reshape_572(%cst_573) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %917 = tensor.empty() : tensor<1x12x169x64xf32>
    %918 = hip.transpose(%arg0) ins(%reshape_574 : tensor<1x169x12x64xf32>) outs(%917 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_575 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_576 = tensor.reshape %910(%cst_575) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %919 = tensor.empty() : tensor<169x768xf32>
    %920 = hip.gemm(%arg0) ins(%reshape_576, %13, %12 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%919 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_577 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_578 = tensor.reshape %920(%cst_577) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_579 = arith.constant dense<[1, 169, -1, 64]> : tensor<4xi64>
    %reshape_580 = tensor.reshape %reshape_578(%cst_579) : (tensor<1x169x768xf32>, tensor<4xi64>) -> tensor<1x169x12x64xf32>
    %921 = tensor.empty() : tensor<1x12x169x64xf32>
    %922 = hip.transpose(%arg0) ins(%reshape_580 : tensor<1x169x12x64xf32>) outs(%921 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %cst_581 = arith.constant 0.000000e+00 : f32
    %cst_582 = arith.constant 0xFF800000 : f32
    %923 = hip.constant {serialization_order = 285 : i64, value = dense<0.000000e+00> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_583 = tensor.splat %cst_581 : tensor<1x1x169x169xf32>
    %924 = hip.constant {serialization_order = 286 : i64, value = dense<0xFF800000> : tensor<1x1x169x169xf32>} : tensor<1x1x169x169xf32>
    %splat_584 = tensor.splat %cst_582 : tensor<1x1x169x169xf32>
    %925 = tensor.empty() : tensor<1x1x169x169xf32>
    %926 = hip.where(%arg0) ins(%319, %923, %924 : tensor<1x1x169x169xi1>, tensor<1x1x169x169xf32>, tensor<1x1x169x169xf32>) outs(%925 : tensor<1x1x169x169xf32>) : tensor<1x1x169x169xf32>
    %927 = tensor.empty() : tensor<1x169x12x64xf32>
    %928 = hip.transpose(%arg0) ins(%914 : tensor<1x12x169x64xf32>) outs(%927 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_585 = tensor.collapse_shape %928 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %929 = tensor.empty() : tensor<1x169x12x64xf32>
    %930 = hip.transpose(%arg0) ins(%918 : tensor<1x12x169x64xf32>) outs(%929 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_586 = tensor.collapse_shape %930 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %931 = tensor.empty() : tensor<1x169x12x64xf32>
    %932 = hip.transpose(%arg0) ins(%922 : tensor<1x12x169x64xf32>) outs(%931 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %collapsed_587 = tensor.collapse_shape %932 [[0], [1], [2, 3]] : tensor<1x169x12x64xf32> into tensor<1x169x768xf32>
    %933 = hip.constant {serialization_order = 287 : i64, value = dense<168> : tensor<1xi32>} : tensor<1xi32>
    %cst_588 = arith.constant dense<168> : tensor<1xi32>
    %934 = hip.constant {serialization_order = 288 : i64, value = dense<169> : tensor<i32>} : tensor<i32>
    %cst_589 = arith.constant dense<169> : tensor<i32>
    %935 = tensor.empty() : tensor<1x169x768xf32>
    %936 = tensor.empty() : tensor<1x12x169x64xf32>
    %937 = tensor.empty() : tensor<1x12x169x64xf32>
    %938:3 = hip.gqa(%arg0) ins(%collapsed_585, %collapsed_586, %collapsed_587, %933, %934, %926 : tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1x169x768xf32>, tensor<1xi32>, tensor<i32>, tensor<1x1x169x169xf32>) outs(%935, %936, %937 : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>) {kv_num_heads = 12 : i64, no_causal = true, num_heads = 12 : i64, scale = 1.250000e-01 : f32} : tensor<1x169x768xf32>, tensor<1x12x169x64xf32>, tensor<1x12x169x64xf32>
    %expanded_590 = tensor.expand_shape %938#0 [[0], [1], [2, 3]] output_shape [1, 169, 12, 64] : tensor<1x169x768xf32> into tensor<1x169x12x64xf32>
    %939 = tensor.empty() : tensor<1x12x169x64xf32>
    %940 = hip.transpose(%arg0) ins(%expanded_590 : tensor<1x169x12x64xf32>) outs(%939 : tensor<1x12x169x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x12x169x64xf32>
    %941 = tensor.empty() : tensor<1x169x12x64xf32>
    %942 = hip.transpose(%arg0) ins(%940 : tensor<1x12x169x64xf32>) outs(%941 : tensor<1x169x12x64xf32>) {perm = [0, 2, 1, 3]} : tensor<1x169x12x64xf32>
    %cst_591 = arith.constant dense<[1, 169, -1]> : tensor<3xi64>
    %reshape_592 = tensor.reshape %942(%cst_591) : (tensor<1x169x12x64xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %cst_593 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_594 = tensor.reshape %reshape_592(%cst_593) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %943 = tensor.empty() : tensor<169x768xf32>
    %944 = hip.gemm(%arg0) ins(%reshape_594, %11, %10 : tensor<169x768xf32>, tensor<768x768xf32>, tensor<768xf32>) outs(%943 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_595 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_596 = tensor.reshape %944(%cst_595) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %945 = tensor.empty() : tensor<1x169x768xf32>
    %946 = hip.add(%arg0) ins(%910, %reshape_596 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%945 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_597 = arith.constant dense<768> : tensor<1xi64>
    %947 = tensor.empty() : tensor<1x169x768xf32>
    %948 = hip.layer_norm(%arg0) ins(%946, %9, %8 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%947 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_598 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_599 = tensor.reshape %948(%cst_598) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %949 = tensor.empty() : tensor<169x3072xf32>
    %950 = hip.gemm(%arg0) ins(%reshape_599, %7, %6 : tensor<169x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) outs(%949 : tensor<169x3072xf32>) {transB = 1 : i64} : tensor<169x3072xf32>
    %cst_600 = arith.constant dense<[1, 169, 3072]> : tensor<3xi64>
    %reshape_601 = tensor.reshape %950(%cst_600) : (tensor<169x3072xf32>, tensor<3xi64>) -> tensor<1x169x3072xf32>
    %951 = tensor.empty() : tensor<1x169x3072xf32>
    %952 = hip.gelu(%arg0) ins(%reshape_601 : tensor<1x169x3072xf32>) outs(%951 : tensor<1x169x3072xf32>) : tensor<1x169x3072xf32>
    %cst_602 = arith.constant dense<[169, 3072]> : tensor<2xi64>
    %reshape_603 = tensor.reshape %952(%cst_602) : (tensor<1x169x3072xf32>, tensor<2xi64>) -> tensor<169x3072xf32>
    %953 = tensor.empty() : tensor<169x768xf32>
    %954 = hip.gemm(%arg0) ins(%reshape_603, %5, %4 : tensor<169x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) outs(%953 : tensor<169x768xf32>) {transB = 1 : i64} : tensor<169x768xf32>
    %cst_604 = arith.constant dense<[1, 169, 768]> : tensor<3xi64>
    %reshape_605 = tensor.reshape %954(%cst_604) : (tensor<169x768xf32>, tensor<3xi64>) -> tensor<1x169x768xf32>
    %955 = tensor.empty() : tensor<1x169x768xf32>
    %956 = hip.add(%arg0) ins(%948, %reshape_605 : tensor<1x169x768xf32>, tensor<1x169x768xf32>) outs(%955 : tensor<1x169x768xf32>) -> tensor<1x169x768xf32>
    %cst_606 = arith.constant dense<768> : tensor<1xi64>
    %957 = tensor.empty() : tensor<1x169x768xf32>
    %958 = hip.layer_norm(%arg0) ins(%956, %3, %2 : tensor<1x169x768xf32>, tensor<768xf32>, tensor<768xf32>) outs(%957 : tensor<1x169x768xf32>) {axis = 2 : i64} : tensor<1x169x768xf32>
    %cst_607 = arith.constant dense<[169, 768]> : tensor<2xi64>
    %reshape_608 = tensor.reshape %958(%cst_607) : (tensor<1x169x768xf32>, tensor<2xi64>) -> tensor<169x768xf32>
    %959 = tensor.empty() : tensor<169x32xf32>
    %960 = hip.gemm(%arg0) ins(%reshape_608, %1, %0 : tensor<169x768xf32>, tensor<32x768xf32>, tensor<32xf32>) outs(%959 : tensor<169x32xf32>) {transB = 1 : i64} : tensor<169x32xf32>
    %cst_609 = arith.constant dense<[1, 169, 32]> : tensor<3xi64>
    %reshape_610 = tensor.reshape %960(%cst_609) : (tensor<169x32xf32>, tensor<3xi64>) -> tensor<1x169x32xf32>
    return %reshape_610 : tensor<1x169x32xf32>
  }
}
