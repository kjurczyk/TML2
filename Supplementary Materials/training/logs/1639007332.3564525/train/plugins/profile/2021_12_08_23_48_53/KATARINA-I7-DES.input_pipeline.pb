	`??"?	6@`??"?	6@!`??"?	6@	Z??%,?1@Z??%,?1@!Z??%,?1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`??"?	6@?9#J{???A??@???1@Yd;?O??@*	ffff?x?@2P
Iterator::Model::Prefetch??_??@!zG?}Y?F@)??_??@1zG?}Y?F@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap6?;N?@!?'??y?J@)??#???@1[]?J???@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?߾?3??!?iW??5@)?߾?3??1?iW??5@:Preprocessing2F
Iterator::ModelO@a?S@!1?|?7G@)L7?A`???1?-??'??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!??D?0ۅ?)??H?}M?1??D?0ۅ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 17.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9Z??%,?1@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?9#J{????9#J{???!?9#J{???      ??!       "      ??!       *      ??!       2	??@???1@??@???1@!??@???1@:      ??!       B      ??!       J	d;?O??@d;?O??@!d;?O??@R      ??!       Z	d;?O??@d;?O??@!d;?O??@JCPU_ONLYYZ??%,?1@b 