	[B>??,#@[B>??,#@![B>??,#@	AP?D?F@AP?D?F@!AP?D?F@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$[B>??,#@?(??0??A?v???@Y??N@?@*	ffff?8?@2P
Iterator::Model::Prefetch!?rh?m@!P??X?IG@)!?rh?m@1P??X?IG@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?J?4?@!'?RmRJ@)?t?V@1aK???A@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?0?*???!??C??!2@)?0?*???1??C??!2@:Preprocessing2F
Iterator::ModelNbX9?@!?(????G@)D?l?????1^"?qN
??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!?s?Z6???)??H?}M?1?s?Z6???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 44.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9AP?D?F@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(??0???(??0??!?(??0??      ??!       "      ??!       *      ??!       2	?v???@?v???@!?v???@:      ??!       B      ??!       J	??N@?@??N@?@!??N@?@R      ??!       Z	??N@?@??N@?@!??N@?@JCPU_ONLYYAP?D?F@b 