	???ׁ?.@???ׁ?.@!???ׁ?.@	?j???G<@?j???G<@!?j???G<@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???ׁ?.@?Q?????A^?I?%@Yı.n??@*	3333s??@2P
Iterator::Model::Prefetch?Zd?@!?95?f?G@)?Zd?@1?95?f?G@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?,C??@!????I@),e?X@1???:@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorc?=yX??!(>5W?3@)c?=yX??1(>5W?3@:Preprocessing2F
Iterator::Model?L?Jj@!??a5'H@)$(~????1?#7{?9??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor/n??R?!????????)/n??R?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 28.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?j???G<@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q??????Q?????!?Q?????      ??!       "      ??!       *      ??!       2	^?I?%@^?I?%@!^?I?%@:      ??!       B      ??!       J	ı.n??@ı.n??@!ı.n??@R      ??!       Z	ı.n??@ı.n??@!ı.n??@JCPU_ONLYY?j???G<@b 