	?m4???@?m4???@!?m4???@	$?>E???$?>E???!$?>E???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?m4???@q=
ףp??A?O??n@Ylxz?,C??*	gffff??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator`??"????!G$??Q?W@)`??"????1G$??Q?W@:Preprocessing2F
Iterator::Model?ݓ??Z??!?I?U??@)???{????1}R9S??@:Preprocessing2P
Iterator::Model::Prefetch??_?Lu?!\??8???)??_?Lu?1\??8???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapu?V??!e??ZP?W@){?G?zd?1??¿???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9$?>E???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	q=
ףp??q=
ףp??!q=
ףp??      ??!       "      ??!       *      ??!       2	?O??n@?O??n@!?O??n@:      ??!       B      ??!       J	lxz?,C??lxz?,C??!lxz?,C??R      ??!       Z	lxz?,C??lxz?,C??!lxz?,C??JCPU_ONLYY$?>E???b 