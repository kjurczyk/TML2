	?H.?!}@?H.?!}@!?H.?!}@	f??p?M@f??p?M@!f??p?M@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?H.?!}@??|гY??AD?????@YB?f???@*	???????@2P
Iterator::Model::PrefetchF%u?@!??D?fI@)F%u?@1??D?fI@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapk+??ݓ
@!I?)?H@)
ףp=
@1????eC@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??_vO??!)?H?
$@)??_vO??1)?H?
$@:Preprocessing2F
Iterator::Model??<,Ԛ@!??F??I@)?4?8EG??1???Fҏ??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?!???̰??)????MbP?1???̰??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 59.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9f??p?M@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??|гY????|гY??!??|гY??      ??!       "      ??!       *      ??!       2	D?????@D?????@!D?????@:      ??!       B      ??!       J	B?f???@B?f???@!B?f???@R      ??!       Z	B?f???@B?f???@!B?f???@JCPU_ONLYYf??p?M@b 