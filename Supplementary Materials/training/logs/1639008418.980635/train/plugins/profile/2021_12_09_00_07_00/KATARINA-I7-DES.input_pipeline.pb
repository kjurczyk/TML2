	?{??Pk???{??Pk??!?{??Pk??	??֜??@??֜??@!??֜??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?{??Pk??q???h??A???QI???Y?H?}??*	?????x@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?;Nё\??!I?Q6ѽV@)?;Nё\??1I?Q6ѽV@:Preprocessing2F
Iterator::Model?St$????!?#䡃I!@)B>?٬???1C???x@:Preprocessing2P
Iterator::Model::Prefetchn??t?!??PNi??)n??t?1??PNi??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapL?
F%u??!?{Ë??V@)?~j?t?X?17?qU???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??֜??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	q???h??q???h??!q???h??      ??!       "      ??!       *      ??!       2	???QI??????QI???!???QI???:      ??!       B      ??!       J	?H?}???H?}??!?H?}??R      ??!       Z	?H?}???H?}??!?H?}??JCPU_ONLYY??֜??@b 