	гY??Z@гY??Z@!гY??Z@	?A?~??@?A?~??@!?A?~??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$гY??Z@??e?c]??AI??&??Y|a2U0??*	?????ف@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator
ףp=
??!?b>?bNW@)
ףp=
??1?b>?bNW@:Preprocessing2F
Iterator::ModelP?s???!?J??J?@)a??+e??19?]9?]@:Preprocessing2P
Iterator::Model::Prefetch/?$???!jiiiii??)/?$???1jiiiii??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap???&??!W{?V{tW@)_?Q?k?1?0?0??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?A?~??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??e?c]????e?c]??!??e?c]??      ??!       "      ??!       *      ??!       2	I??&??I??&??!I??&??:      ??!       B      ??!       J	|a2U0??|a2U0??!|a2U0??R      ??!       Z	|a2U0??|a2U0??!|a2U0??JCPU_ONLYY?A?~??@b 