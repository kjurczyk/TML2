	??H.?A*@??H.?A*@!??H.?A*@	?[?^{jC@?[?^{jC@!?[?^{jC@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??H.?A*@'?W???AӼ??@Yj?q??d@*	43333??@2P
Iterator::Model::Prefetch?|гY?@!??$??F@)?|гY?@1??$??F@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapt???.@!?&4y?J@)b??4??@1Zr-????@:Preprocessing2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?A`?Т@!5#???i5@)?A`?Т@15#???i5@:Preprocessing2F
Iterator::ModelM?O?T@!D??ˆ\G@)?A?fշ?1?I???b??:Preprocessing2e
.Iterator::Model::Prefetch::FlatMap::FromTensor??H?}M?!?H?<????)??H?}M?1?H?<????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 38.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?[?^{jC@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'?W???'?W???!'?W???      ??!       "      ??!       *      ??!       2	Ӽ??@Ӽ??@!Ӽ??@:      ??!       B      ??!       J	j?q??d@j?q??d@!j?q??d@R      ??!       Z	j?q??d@j?q??d@!j?q??d@JCPU_ONLYY?[?^{jC@b 