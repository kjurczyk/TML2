	r?????@r?????@!r?????@	? ?z?R@? ?z?R@!? ?z?R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$r?????@?W[?????A??Pk?w??Y??镲@*	3333???@2P
Iterator::Model::Prefetchf?c]?F@!??ϩI@)f?c]?F@1??ϩI@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?A?f?g
@!????IH@)?E????@1J?&??D@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorlxz?,C??!Yn?c? @)lxz?,C??1Yn?c? @:Preprocessing2F
Iterator::Model?ʡE??@!<?B?I@)^K?=???1??????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?!?H~?!$??)????MbP?1?H~?!$??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 74.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9? ?z?R@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?W[??????W[?????!?W[?????      ??!       "      ??!       *      ??!       2	??Pk?w????Pk?w??!??Pk?w??:      ??!       B      ??!       J	??镲@??镲@!??镲@R      ??!       Z	??镲@??镲@!??镲@JCPU_ONLYY? ?z?R@b 