	??9#J;(@??9#J;(@!??9#J;(@	]?&??xA@]?&??xA@!]?&??xA@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??9#J;(@??(????A7?[ @Y:#J{??@*	???̌K?@2P
Iterator::Model::Prefetchc?=y?@!????6mG@)c?=y?@1????6mG@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?????@!(?S)o+J@)B?f???@1????,@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorvOjM??!PG? ??3@)vOjM??1PG? ??3@:Preprocessing2F
Iterator::ModelR'????@!?]?֐?G@)?????M??1_vl????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor/n??R?!uz?#?p??)/n??R?1uz?#?p??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9]?&??xA@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??(??????(????!??(????      ??!       "      ??!       *      ??!       2	7?[ @7?[ @!7?[ @:      ??!       B      ??!       J	:#J{??@:#J{??@!:#J{??@R      ??!       Z	:#J{??@:#J{??@!:#J{??@JCPU_ONLYY]?&??xA@b 