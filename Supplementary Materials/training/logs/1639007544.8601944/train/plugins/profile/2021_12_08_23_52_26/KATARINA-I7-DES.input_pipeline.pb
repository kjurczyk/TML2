	?0?*(@?0?*(@!?0?*(@	???!?B@???!?B@!???!?B@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?0?*(@???(???A??a?ִ@Yv????@*	????ِ?@2P
Iterator::Model::Prefetch ?~?:?@!r???BG@) ?~?:?@1r???BG@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapjM??@!?C	??fJ@))\????@1?"????@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?HP| @!;e?ᶭ5@)?HP| @1;e?ᶭ5@:Preprocessing2F
Iterator::Model5?8EG?@!U??oz?G@)P??n???1???????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!R????c??)??H?}M?1R????c??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 37.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???!?B@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???(??????(???!???(???      ??!       "      ??!       *      ??!       2	??a?ִ@??a?ִ@!??a?ִ@:      ??!       B      ??!       J	v????@v????@!v????@R      ??!       Z	v????@v????@!v????@JCPU_ONLYY???!?B@b 