	??ׁsF@??ׁsF@!??ׁsF@	Gȴ??S@Gȴ??S@!Gȴ??S@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ׁsF@?<,Ԛ???A4??@????Y?&1??@*	????? ?@2P
Iterator::Model::Prefetch?z?G?@!\???$5I@)?z?G?@1\???$5I@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapvOj?
@!?l???;H@)?o_Ι@12.?j6oD@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatoro???T???!??*]L@)o???T???1??*]L@:Preprocessing2F
Iterator::Model?St$?@!,?v}G?I@)????ɳ?1:?W???:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor-C??6Z?!?G????)-C??6Z?1?G????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 78.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9Gȴ??S@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?<,Ԛ????<,Ԛ???!?<,Ԛ???      ??!       "      ??!       *      ??!       2	4??@????4??@????!4??@????:      ??!       B      ??!       J	?&1??@?&1??@!?&1??@R      ??!       Z	?&1??@?&1??@!?&1??@JCPU_ONLYYGȴ??S@b 