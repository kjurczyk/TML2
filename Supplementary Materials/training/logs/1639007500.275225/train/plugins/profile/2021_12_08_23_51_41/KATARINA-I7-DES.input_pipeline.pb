	 c?ZB?!@ c?ZB?!@! c?ZB?!@	tz???E@tz???E@!tz???E@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ c?ZB?!@??o_??AJ+?V@Y?Q??k@*	3333??@2P
Iterator::Model::PrefetchȘ????@!?"??'KG@)Ș????@1?"??'KG@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapJ{?/L&@!\l%Û$J@)Zd;??@1??I?@@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?|гY???!>	???3@)?|гY???1>	???3@:Preprocessing2F
Iterator::Model?????L@!???<d?G@)??&???1QN????:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor????MbP?!?˯?????)????MbP?1?˯?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 44.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9tz???E@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??o_????o_??!??o_??      ??!       "      ??!       *      ??!       2	J+?V@J+?V@!J+?V@:      ??!       B      ??!       J	?Q??k@?Q??k@!?Q??k@R      ??!       Z	?Q??k@?Q??k@!?Q??k@JCPU_ONLYYtz???E@b 