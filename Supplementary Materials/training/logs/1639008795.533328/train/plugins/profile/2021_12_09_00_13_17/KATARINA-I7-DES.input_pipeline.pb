	?.n??@?.n??@!?.n??@	??t?XjO@??t?XjO@!??t?XjO@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?.n??@??N@a??A?MbX9@Yk?w??#@*	     2?@2P
Iterator::Model::Prefetch??6?[@!
-M??kI@)??6?[@1
-M??kI@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?=yX??@!
?,CU?G@)lxz?,C@1?E??FC@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator???????!\????"@)???????1\????"@:Preprocessing2F
Iterator::Model?:p?@!?aӼ?J@)o??ʡ??1U?Ɛ???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?! ?&e?_??)????MbP?1 ?&e?_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 62.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9??t?XjO@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??N@a????N@a??!??N@a??      ??!       "      ??!       *      ??!       2	?MbX9@?MbX9@!?MbX9@:      ??!       B      ??!       J	k?w??#@k?w??#@!k?w??#@R      ??!       Z	k?w??#@k?w??#@!k?w??#@JCPU_ONLYY??t?XjO@b 