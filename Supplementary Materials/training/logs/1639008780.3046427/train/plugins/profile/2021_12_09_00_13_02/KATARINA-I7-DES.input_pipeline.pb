	??	h?@??	h?@!??	h?@	???/?$R@???/?$R@!???/?$R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??	h?@+??	h??A?|a2U??Y?ʡE??@*	3333???@2P
Iterator::Model::Prefetch䃞ͪO@!ŋYuI@)䃞ͪO@1ŋYuI@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?3???@!??&???H@).?!???@1YgR??C@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?H.?!???!d?(??#@)?H.?!???1d?(??#@:Preprocessing2F
Iterator::Modelw??/?@!?P>}I@)TR'?????1????N2??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensora2U0*?S?!j?w??ǐ?)a2U0*?S?1j?w??ǐ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 72.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???/?$R@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+??	h??+??	h??!+??	h??      ??!       "      ??!       *      ??!       2	?|a2U???|a2U??!?|a2U??:      ??!       B      ??!       J	?ʡE??@?ʡE??@!?ʡE??@R      ??!       Z	?ʡE??@?ʡE??@!?ʡE??@JCPU_ONLYY???/?$R@b 