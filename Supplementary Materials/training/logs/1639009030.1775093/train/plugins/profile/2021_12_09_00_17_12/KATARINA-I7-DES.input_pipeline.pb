	z?):?@z?):?@!z?):?@	???,~N@???,~N@!???,~N@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$z?):?@???{????Az6?>W@Y|??Pk@*	gfff&?@2P
Iterator::Model::Prefetch?c]?F?@!oکE1?I@)?c]?F?@1oکE1?I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap??k	??@!?Iݽ?H@)u??@1?V'o?B@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?
F%u??!?e?O%@)?
F%u??1?e?O%@:Preprocessing2F
Iterator::Model?D???
@!6?"BE?I@)?Q?????1??6?D??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor/n??R?!X?Q?l??)/n??R?1X?Q?l??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 61.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???,~N@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???{???????{????!???{????      ??!       "      ??!       *      ??!       2	z6?>W@z6?>W@!z6?>W@:      ??!       B      ??!       J	|??Pk@|??Pk@!|??Pk@R      ??!       Z	|??Pk@|??Pk@!|??Pk@JCPU_ONLYY???,~N@b 