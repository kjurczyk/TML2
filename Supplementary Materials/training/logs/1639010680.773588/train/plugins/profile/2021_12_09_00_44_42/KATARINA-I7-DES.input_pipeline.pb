	?q???@?q???@!?q???@	???%?????%??!???%??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?q???@?D???J??A?n???@Y"??u????*	fffff?z@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator=
ףp=??!̜?3??W@)=
ףp=??1̜?3??W@:Preprocessing2F
Iterator::Model??y?):??!|$Z?Δ@)a??+e??1,???@:Preprocessing2P
Iterator::Model::Prefetch??_vOv?!?ǒ[ ??)??_vOv?1?ǒ[ ??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?&?W??!?]???W@)-C??6Z?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???%??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?D???J???D???J??!?D???J??      ??!       "      ??!       *      ??!       2	?n???@?n???@!?n???@:      ??!       B      ??!       J	"??u????"??u????!"??u????R      ??!       Z	"??u????"??u????!"??u????JCPU_ONLYY???%??b 