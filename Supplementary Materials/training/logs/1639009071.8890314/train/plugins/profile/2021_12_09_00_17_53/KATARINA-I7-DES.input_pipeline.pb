	\???(\!@\???(\!@!\???(\!@	?Tx*<uT@?Tx*<uT@!?Tx*<uT@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$\???(\!@RI??&¶?As??A???Y????xi@*	3333sv?@2P
Iterator::Model::Prefetch$(~??@!v&??!?I@)$(~??@1v&??!?I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapгY???@!?b?T?G@)NbX9t@1?47^8?D@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?HP???!Q?K??@)?HP???1Q?K??@:Preprocessing2F
Iterator::Model,e?XW@!9??	?%J@)HP?s??1??]?\???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor_?Q?{?!?z?*Q???)_?Q?{?1?z?*Q???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 81.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?Tx*<uT@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	RI??&¶?RI??&¶?!RI??&¶?      ??!       "      ??!       *      ??!       2	s??A???s??A???!s??A???:      ??!       B      ??!       J	????xi@????xi@!????xi@R      ??!       Z	????xi@????xi@!????xi@JCPU_ONLYY?Tx*<uT@b 