	a2U0*?!@a2U0*?!@!a2U0*?!@	m}?["G@m}?["G@!m}?["G@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$a2U0*?!@????ɳ?A=?U??@Yd;?O?W@*	    ?w?@2P
Iterator::Model::Prefetch??s??@!{5c?ҴG@)??s??@1{5c?ҴG@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?[ Aq@!?(1??I@)$????[@1 ?`?B@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorF%u???!Jg@?./@)F%u???1Jg@?./@:Preprocessing2F
Iterator::ModelF%u?H@!???$H@)????ò?1ץ,?????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor/n??R?!??????)/n??R?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 46.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9m}?["G@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????ɳ?????ɳ?!????ɳ?      ??!       "      ??!       *      ??!       2	=?U??@=?U??@!=?U??@:      ??!       B      ??!       J	d;?O?W@d;?O?W@!d;?O?W@R      ??!       Z	d;?O?W@d;?O?W@!d;?O?W@JCPU_ONLYYm}?["G@b 