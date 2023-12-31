��)
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��#
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
�
Adam/v/PredictionMask/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/v/PredictionMask/bias
�
.Adam/v/PredictionMask/bias/Read/ReadVariableOpReadVariableOpAdam/v/PredictionMask/bias*
_output_shapes
:*
dtype0
�
Adam/m/PredictionMask/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m/PredictionMask/bias
�
.Adam/m/PredictionMask/bias/Read/ReadVariableOpReadVariableOpAdam/m/PredictionMask/bias*
_output_shapes
:*
dtype0
�
Adam/v/PredictionMask/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/v/PredictionMask/kernel
�
0Adam/v/PredictionMask/kernel/Read/ReadVariableOpReadVariableOpAdam/v/PredictionMask/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/PredictionMask/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/m/PredictionMask/kernel
�
0Adam/m/PredictionMask/kernel/Read/ReadVariableOpReadVariableOpAdam/m/PredictionMask/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/convOutb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/convOutb/bias
y
(Adam/v/convOutb/bias/Read/ReadVariableOpReadVariableOpAdam/v/convOutb/bias*
_output_shapes
:*
dtype0
�
Adam/m/convOutb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/convOutb/bias
y
(Adam/m/convOutb/bias/Read/ReadVariableOpReadVariableOpAdam/m/convOutb/bias*
_output_shapes
:*
dtype0
�
Adam/v/convOutb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/convOutb/kernel
�
*Adam/v/convOutb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/convOutb/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/convOutb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/convOutb/kernel
�
*Adam/m/convOutb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/convOutb/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/convOuta/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/convOuta/bias
y
(Adam/v/convOuta/bias/Read/ReadVariableOpReadVariableOpAdam/v/convOuta/bias*
_output_shapes
:*
dtype0
�
Adam/m/convOuta/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/convOuta/bias
y
(Adam/m/convOuta/bias/Read/ReadVariableOpReadVariableOpAdam/m/convOuta/bias*
_output_shapes
:*
dtype0
�
Adam/v/convOuta/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/convOuta/kernel
�
*Adam/v/convOuta/kernel/Read/ReadVariableOpReadVariableOpAdam/v/convOuta/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/convOuta/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/convOuta/kernel
�
*Adam/m/convOuta/kernel/Read/ReadVariableOpReadVariableOpAdam/m/convOuta/kernel*&
_output_shapes
: *
dtype0
�
Adam/v/transconvA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/transconvA/bias
}
*Adam/v/transconvA/bias/Read/ReadVariableOpReadVariableOpAdam/v/transconvA/bias*
_output_shapes
:*
dtype0
�
Adam/m/transconvA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/transconvA/bias
}
*Adam/m/transconvA/bias/Read/ReadVariableOpReadVariableOpAdam/m/transconvA/bias*
_output_shapes
:*
dtype0
�
Adam/v/transconvA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/v/transconvA/kernel
�
,Adam/v/transconvA/kernel/Read/ReadVariableOpReadVariableOpAdam/v/transconvA/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/transconvA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/m/transconvA/kernel
�
,Adam/m/transconvA/kernel/Read/ReadVariableOpReadVariableOpAdam/m/transconvA/kernel*&
_output_shapes
: *
dtype0
�
Adam/v/decodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/decodeAb/bias
y
(Adam/v/decodeAb/bias/Read/ReadVariableOpReadVariableOpAdam/v/decodeAb/bias*
_output_shapes
: *
dtype0
�
Adam/m/decodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/decodeAb/bias
y
(Adam/m/decodeAb/bias/Read/ReadVariableOpReadVariableOpAdam/m/decodeAb/bias*
_output_shapes
: *
dtype0
�
Adam/v/decodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/v/decodeAb/kernel
�
*Adam/v/decodeAb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/decodeAb/kernel*&
_output_shapes
:  *
dtype0
�
Adam/m/decodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/m/decodeAb/kernel
�
*Adam/m/decodeAb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/decodeAb/kernel*&
_output_shapes
:  *
dtype0
�
Adam/v/decodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/decodeAa/bias
y
(Adam/v/decodeAa/bias/Read/ReadVariableOpReadVariableOpAdam/v/decodeAa/bias*
_output_shapes
: *
dtype0
�
Adam/m/decodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/decodeAa/bias
y
(Adam/m/decodeAa/bias/Read/ReadVariableOpReadVariableOpAdam/m/decodeAa/bias*
_output_shapes
: *
dtype0
�
Adam/v/decodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/v/decodeAa/kernel
�
*Adam/v/decodeAa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/decodeAa/kernel*&
_output_shapes
:@ *
dtype0
�
Adam/m/decodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/m/decodeAa/kernel
�
*Adam/m/decodeAa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/decodeAa/kernel*&
_output_shapes
:@ *
dtype0
�
Adam/v/transconvB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/transconvB/bias
}
*Adam/v/transconvB/bias/Read/ReadVariableOpReadVariableOpAdam/v/transconvB/bias*
_output_shapes
: *
dtype0
�
Adam/m/transconvB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/transconvB/bias
}
*Adam/m/transconvB/bias/Read/ReadVariableOpReadVariableOpAdam/m/transconvB/bias*
_output_shapes
: *
dtype0
�
Adam/v/transconvB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/v/transconvB/kernel
�
,Adam/v/transconvB/kernel/Read/ReadVariableOpReadVariableOpAdam/v/transconvB/kernel*&
_output_shapes
: @*
dtype0
�
Adam/m/transconvB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/m/transconvB/kernel
�
,Adam/m/transconvB/kernel/Read/ReadVariableOpReadVariableOpAdam/m/transconvB/kernel*&
_output_shapes
: @*
dtype0
�
Adam/v/decodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/decodeBb/bias
y
(Adam/v/decodeBb/bias/Read/ReadVariableOpReadVariableOpAdam/v/decodeBb/bias*
_output_shapes
:@*
dtype0
�
Adam/m/decodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/decodeBb/bias
y
(Adam/m/decodeBb/bias/Read/ReadVariableOpReadVariableOpAdam/m/decodeBb/bias*
_output_shapes
:@*
dtype0
�
Adam/v/decodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/decodeBb/kernel
�
*Adam/v/decodeBb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/decodeBb/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/m/decodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/decodeBb/kernel
�
*Adam/m/decodeBb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/decodeBb/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/v/decodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/decodeBa/bias
y
(Adam/v/decodeBa/bias/Read/ReadVariableOpReadVariableOpAdam/v/decodeBa/bias*
_output_shapes
:@*
dtype0
�
Adam/m/decodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/decodeBa/bias
y
(Adam/m/decodeBa/bias/Read/ReadVariableOpReadVariableOpAdam/m/decodeBa/bias*
_output_shapes
:@*
dtype0
�
Adam/v/decodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*'
shared_nameAdam/v/decodeBa/kernel
�
*Adam/v/decodeBa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/decodeBa/kernel*'
_output_shapes
:�@*
dtype0
�
Adam/m/decodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*'
shared_nameAdam/m/decodeBa/kernel
�
*Adam/m/decodeBa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/decodeBa/kernel*'
_output_shapes
:�@*
dtype0
�
Adam/v/transconvC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/v/transconvC/bias
}
*Adam/v/transconvC/bias/Read/ReadVariableOpReadVariableOpAdam/v/transconvC/bias*
_output_shapes
:@*
dtype0
�
Adam/m/transconvC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/m/transconvC/bias
}
*Adam/m/transconvC/bias/Read/ReadVariableOpReadVariableOpAdam/m/transconvC/bias*
_output_shapes
:@*
dtype0
�
Adam/v/transconvC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*)
shared_nameAdam/v/transconvC/kernel
�
,Adam/v/transconvC/kernel/Read/ReadVariableOpReadVariableOpAdam/v/transconvC/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/m/transconvC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*)
shared_nameAdam/m/transconvC/kernel
�
,Adam/m/transconvC/kernel/Read/ReadVariableOpReadVariableOpAdam/m/transconvC/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/v/decodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/decodeCb/bias
z
(Adam/v/decodeCb/bias/Read/ReadVariableOpReadVariableOpAdam/v/decodeCb/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/decodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/decodeCb/bias
z
(Adam/m/decodeCb/bias/Read/ReadVariableOpReadVariableOpAdam/m/decodeCb/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/decodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/v/decodeCb/kernel
�
*Adam/v/decodeCb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/decodeCb/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/decodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/m/decodeCb/kernel
�
*Adam/m/decodeCb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/decodeCb/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/decodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/decodeCa/bias
z
(Adam/v/decodeCa/bias/Read/ReadVariableOpReadVariableOpAdam/v/decodeCa/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/decodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/decodeCa/bias
z
(Adam/m/decodeCa/bias/Read/ReadVariableOpReadVariableOpAdam/m/decodeCa/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/decodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/v/decodeCa/kernel
�
*Adam/v/decodeCa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/decodeCa/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/decodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/m/decodeCa/kernel
�
*Adam/m/decodeCa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/decodeCa/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/transconvE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/v/transconvE/bias
~
*Adam/v/transconvE/bias/Read/ReadVariableOpReadVariableOpAdam/v/transconvE/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/transconvE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/m/transconvE/bias
~
*Adam/m/transconvE/bias/Read/ReadVariableOpReadVariableOpAdam/m/transconvE/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/transconvE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*)
shared_nameAdam/v/transconvE/kernel
�
,Adam/v/transconvE/kernel/Read/ReadVariableOpReadVariableOpAdam/v/transconvE/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/transconvE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*)
shared_nameAdam/m/transconvE/kernel
�
,Adam/m/transconvE/kernel/Read/ReadVariableOpReadVariableOpAdam/m/transconvE/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/encodeEb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/encodeEb/bias
z
(Adam/v/encodeEb/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeEb/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/encodeEb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/encodeEb/bias
z
(Adam/m/encodeEb/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeEb/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/encodeEb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/v/encodeEb/kernel
�
*Adam/v/encodeEb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeEb/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/encodeEb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/m/encodeEb/kernel
�
*Adam/m/encodeEb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeEb/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/encodeEa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/encodeEa/bias
z
(Adam/v/encodeEa/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeEa/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/encodeEa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/encodeEa/bias
z
(Adam/m/encodeEa/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeEa/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/encodeEa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/v/encodeEa/kernel
�
*Adam/v/encodeEa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeEa/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/encodeEa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/m/encodeEa/kernel
�
*Adam/m/encodeEa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeEa/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/encodeDb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/encodeDb/bias
z
(Adam/v/encodeDb/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeDb/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/encodeDb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/encodeDb/bias
z
(Adam/m/encodeDb/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeDb/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/encodeDb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/v/encodeDb/kernel
�
*Adam/v/encodeDb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeDb/kernel*(
_output_shapes
:��*
dtype0
�
Adam/m/encodeDb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/m/encodeDb/kernel
�
*Adam/m/encodeDb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeDb/kernel*(
_output_shapes
:��*
dtype0
�
Adam/v/encodeDa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/encodeDa/bias
z
(Adam/v/encodeDa/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeDa/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/encodeDa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/encodeDa/bias
z
(Adam/m/encodeDa/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeDa/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/encodeDa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/v/encodeDa/kernel
�
*Adam/v/encodeDa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeDa/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/m/encodeDa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/m/encodeDa/kernel
�
*Adam/m/encodeDa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeDa/kernel*'
_output_shapes
:@�*
dtype0
�
Adam/v/encodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/encodeCb/bias
y
(Adam/v/encodeCb/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeCb/bias*
_output_shapes
:@*
dtype0
�
Adam/m/encodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/encodeCb/bias
y
(Adam/m/encodeCb/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeCb/bias*
_output_shapes
:@*
dtype0
�
Adam/v/encodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/encodeCb/kernel
�
*Adam/v/encodeCb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeCb/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/m/encodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/encodeCb/kernel
�
*Adam/m/encodeCb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeCb/kernel*&
_output_shapes
:@@*
dtype0
�
Adam/v/encodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/encodeCa/bias
y
(Adam/v/encodeCa/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeCa/bias*
_output_shapes
:@*
dtype0
�
Adam/m/encodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/encodeCa/bias
y
(Adam/m/encodeCa/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeCa/bias*
_output_shapes
:@*
dtype0
�
Adam/v/encodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/v/encodeCa/kernel
�
*Adam/v/encodeCa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeCa/kernel*&
_output_shapes
: @*
dtype0
�
Adam/m/encodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/m/encodeCa/kernel
�
*Adam/m/encodeCa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeCa/kernel*&
_output_shapes
: @*
dtype0
�
Adam/v/encodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/encodeBb/bias
y
(Adam/v/encodeBb/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeBb/bias*
_output_shapes
: *
dtype0
�
Adam/m/encodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/encodeBb/bias
y
(Adam/m/encodeBb/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeBb/bias*
_output_shapes
: *
dtype0
�
Adam/v/encodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/v/encodeBb/kernel
�
*Adam/v/encodeBb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeBb/kernel*&
_output_shapes
:  *
dtype0
�
Adam/m/encodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/m/encodeBb/kernel
�
*Adam/m/encodeBb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeBb/kernel*&
_output_shapes
:  *
dtype0
�
Adam/v/encodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/encodeBa/bias
y
(Adam/v/encodeBa/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeBa/bias*
_output_shapes
: *
dtype0
�
Adam/m/encodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/encodeBa/bias
y
(Adam/m/encodeBa/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeBa/bias*
_output_shapes
: *
dtype0
�
Adam/v/encodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/encodeBa/kernel
�
*Adam/v/encodeBa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeBa/kernel*&
_output_shapes
: *
dtype0
�
Adam/m/encodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/encodeBa/kernel
�
*Adam/m/encodeBa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeBa/kernel*&
_output_shapes
: *
dtype0
�
Adam/v/encodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/encodeAb/bias
y
(Adam/v/encodeAb/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeAb/bias*
_output_shapes
:*
dtype0
�
Adam/m/encodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/encodeAb/bias
y
(Adam/m/encodeAb/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeAb/bias*
_output_shapes
:*
dtype0
�
Adam/v/encodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/encodeAb/kernel
�
*Adam/v/encodeAb/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeAb/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/encodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/encodeAb/kernel
�
*Adam/m/encodeAb/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeAb/kernel*&
_output_shapes
:*
dtype0
�
Adam/v/encodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/encodeAa/bias
y
(Adam/v/encodeAa/bias/Read/ReadVariableOpReadVariableOpAdam/v/encodeAa/bias*
_output_shapes
:*
dtype0
�
Adam/m/encodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/encodeAa/bias
y
(Adam/m/encodeAa/bias/Read/ReadVariableOpReadVariableOpAdam/m/encodeAa/bias*
_output_shapes
:*
dtype0
�
Adam/v/encodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/encodeAa/kernel
�
*Adam/v/encodeAa/kernel/Read/ReadVariableOpReadVariableOpAdam/v/encodeAa/kernel*&
_output_shapes
:*
dtype0
�
Adam/m/encodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/encodeAa/kernel
�
*Adam/m/encodeAa/kernel/Read/ReadVariableOpReadVariableOpAdam/m/encodeAa/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
~
PredictionMask/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namePredictionMask/bias
w
'PredictionMask/bias/Read/ReadVariableOpReadVariableOpPredictionMask/bias*
_output_shapes
:*
dtype0
�
PredictionMask/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namePredictionMask/kernel
�
)PredictionMask/kernel/Read/ReadVariableOpReadVariableOpPredictionMask/kernel*&
_output_shapes
:*
dtype0
r
convOutb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconvOutb/bias
k
!convOutb/bias/Read/ReadVariableOpReadVariableOpconvOutb/bias*
_output_shapes
:*
dtype0
�
convOutb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconvOutb/kernel
{
#convOutb/kernel/Read/ReadVariableOpReadVariableOpconvOutb/kernel*&
_output_shapes
:*
dtype0
r
convOuta/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconvOuta/bias
k
!convOuta/bias/Read/ReadVariableOpReadVariableOpconvOuta/bias*
_output_shapes
:*
dtype0
�
convOuta/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvOuta/kernel
{
#convOuta/kernel/Read/ReadVariableOpReadVariableOpconvOuta/kernel*&
_output_shapes
: *
dtype0
v
transconvA/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nametransconvA/bias
o
#transconvA/bias/Read/ReadVariableOpReadVariableOptransconvA/bias*
_output_shapes
:*
dtype0
�
transconvA/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nametransconvA/kernel

%transconvA/kernel/Read/ReadVariableOpReadVariableOptransconvA/kernel*&
_output_shapes
: *
dtype0
r
decodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecodeAb/bias
k
!decodeAb/bias/Read/ReadVariableOpReadVariableOpdecodeAb/bias*
_output_shapes
: *
dtype0
�
decodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_namedecodeAb/kernel
{
#decodeAb/kernel/Read/ReadVariableOpReadVariableOpdecodeAb/kernel*&
_output_shapes
:  *
dtype0
r
decodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecodeAa/bias
k
!decodeAa/bias/Read/ReadVariableOpReadVariableOpdecodeAa/bias*
_output_shapes
: *
dtype0
�
decodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_namedecodeAa/kernel
{
#decodeAa/kernel/Read/ReadVariableOpReadVariableOpdecodeAa/kernel*&
_output_shapes
:@ *
dtype0
v
transconvB/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nametransconvB/bias
o
#transconvB/bias/Read/ReadVariableOpReadVariableOptransconvB/bias*
_output_shapes
: *
dtype0
�
transconvB/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nametransconvB/kernel

%transconvB/kernel/Read/ReadVariableOpReadVariableOptransconvB/kernel*&
_output_shapes
: @*
dtype0
r
decodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedecodeBb/bias
k
!decodeBb/bias/Read/ReadVariableOpReadVariableOpdecodeBb/bias*
_output_shapes
:@*
dtype0
�
decodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_namedecodeBb/kernel
{
#decodeBb/kernel/Read/ReadVariableOpReadVariableOpdecodeBb/kernel*&
_output_shapes
:@@*
dtype0
r
decodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedecodeBa/bias
k
!decodeBa/bias/Read/ReadVariableOpReadVariableOpdecodeBa/bias*
_output_shapes
:@*
dtype0
�
decodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@* 
shared_namedecodeBa/kernel
|
#decodeBa/kernel/Read/ReadVariableOpReadVariableOpdecodeBa/kernel*'
_output_shapes
:�@*
dtype0
v
transconvC/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nametransconvC/bias
o
#transconvC/bias/Read/ReadVariableOpReadVariableOptransconvC/bias*
_output_shapes
:@*
dtype0
�
transconvC/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*"
shared_nametransconvC/kernel
�
%transconvC/kernel/Read/ReadVariableOpReadVariableOptransconvC/kernel*'
_output_shapes
:@�*
dtype0
s
decodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedecodeCb/bias
l
!decodeCb/bias/Read/ReadVariableOpReadVariableOpdecodeCb/bias*
_output_shapes	
:�*
dtype0
�
decodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_namedecodeCb/kernel
}
#decodeCb/kernel/Read/ReadVariableOpReadVariableOpdecodeCb/kernel*(
_output_shapes
:��*
dtype0
s
decodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedecodeCa/bias
l
!decodeCa/bias/Read/ReadVariableOpReadVariableOpdecodeCa/bias*
_output_shapes	
:�*
dtype0
�
decodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_namedecodeCa/kernel
}
#decodeCa/kernel/Read/ReadVariableOpReadVariableOpdecodeCa/kernel*(
_output_shapes
:��*
dtype0
w
transconvE/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nametransconvE/bias
p
#transconvE/bias/Read/ReadVariableOpReadVariableOptransconvE/bias*
_output_shapes	
:�*
dtype0
�
transconvE/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*"
shared_nametransconvE/kernel
�
%transconvE/kernel/Read/ReadVariableOpReadVariableOptransconvE/kernel*(
_output_shapes
:��*
dtype0
s
encodeEb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeEb/bias
l
!encodeEb/bias/Read/ReadVariableOpReadVariableOpencodeEb/bias*
_output_shapes	
:�*
dtype0
�
encodeEb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameencodeEb/kernel
}
#encodeEb/kernel/Read/ReadVariableOpReadVariableOpencodeEb/kernel*(
_output_shapes
:��*
dtype0
s
encodeEa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeEa/bias
l
!encodeEa/bias/Read/ReadVariableOpReadVariableOpencodeEa/bias*
_output_shapes	
:�*
dtype0
�
encodeEa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameencodeEa/kernel
}
#encodeEa/kernel/Read/ReadVariableOpReadVariableOpencodeEa/kernel*(
_output_shapes
:��*
dtype0
s
encodeDb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeDb/bias
l
!encodeDb/bias/Read/ReadVariableOpReadVariableOpencodeDb/bias*
_output_shapes	
:�*
dtype0
�
encodeDb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameencodeDb/kernel
}
#encodeDb/kernel/Read/ReadVariableOpReadVariableOpencodeDb/kernel*(
_output_shapes
:��*
dtype0
s
encodeDa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameencodeDa/bias
l
!encodeDa/bias/Read/ReadVariableOpReadVariableOpencodeDa/bias*
_output_shapes	
:�*
dtype0
�
encodeDa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameencodeDa/kernel
|
#encodeDa/kernel/Read/ReadVariableOpReadVariableOpencodeDa/kernel*'
_output_shapes
:@�*
dtype0
r
encodeCb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameencodeCb/bias
k
!encodeCb/bias/Read/ReadVariableOpReadVariableOpencodeCb/bias*
_output_shapes
:@*
dtype0
�
encodeCb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameencodeCb/kernel
{
#encodeCb/kernel/Read/ReadVariableOpReadVariableOpencodeCb/kernel*&
_output_shapes
:@@*
dtype0
r
encodeCa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameencodeCa/bias
k
!encodeCa/bias/Read/ReadVariableOpReadVariableOpencodeCa/bias*
_output_shapes
:@*
dtype0
�
encodeCa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameencodeCa/kernel
{
#encodeCa/kernel/Read/ReadVariableOpReadVariableOpencodeCa/kernel*&
_output_shapes
: @*
dtype0
r
encodeBb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameencodeBb/bias
k
!encodeBb/bias/Read/ReadVariableOpReadVariableOpencodeBb/bias*
_output_shapes
: *
dtype0
�
encodeBb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameencodeBb/kernel
{
#encodeBb/kernel/Read/ReadVariableOpReadVariableOpencodeBb/kernel*&
_output_shapes
:  *
dtype0
r
encodeBa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameencodeBa/bias
k
!encodeBa/bias/Read/ReadVariableOpReadVariableOpencodeBa/bias*
_output_shapes
: *
dtype0
�
encodeBa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameencodeBa/kernel
{
#encodeBa/kernel/Read/ReadVariableOpReadVariableOpencodeBa/kernel*&
_output_shapes
: *
dtype0
r
encodeAb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameencodeAb/bias
k
!encodeAb/bias/Read/ReadVariableOpReadVariableOpencodeAb/bias*
_output_shapes
:*
dtype0
�
encodeAb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameencodeAb/kernel
{
#encodeAb/kernel/Read/ReadVariableOpReadVariableOpencodeAb/kernel*&
_output_shapes
:*
dtype0
r
encodeAa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameencodeAa/bias
k
!encodeAa/bias/Read/ReadVariableOpReadVariableOpencodeAa/bias*
_output_shapes
:*
dtype0
�
encodeAa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameencodeAa/kernel
{
#encodeAa/kernel/Read/ReadVariableOpReadVariableOpencodeAa/kernel*&
_output_shapes
:*
dtype0
�
serving_default_MRImagesPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_MRImagesencodeAa/kernelencodeAa/biasencodeAb/kernelencodeAb/biasencodeBa/kernelencodeBa/biasencodeBb/kernelencodeBb/biasencodeCa/kernelencodeCa/biasencodeCb/kernelencodeCb/biasencodeDa/kernelencodeDa/biasencodeDb/kernelencodeDb/biasencodeEa/kernelencodeEa/biasencodeEb/kernelencodeEb/biastransconvE/kerneltransconvE/biasdecodeCa/kerneldecodeCa/biasdecodeCb/kerneldecodeCb/biastransconvC/kerneltransconvC/biasdecodeBa/kerneldecodeBa/biasdecodeBb/kerneldecodeBb/biastransconvB/kerneltransconvB/biasdecodeAa/kerneldecodeAa/biasdecodeAb/kerneldecodeAb/biastransconvA/kerneltransconvA/biasconvOuta/kernelconvOuta/biasconvOutb/kernelconvOutb/biasPredictionMask/kernelPredictionMask/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_191983

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!layer_with_weights-21
!layer-32
"layer_with_weights-22
"layer-33
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_default_save_signature
*	optimizer
+
signatures*

,_init_input_shape* 
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses* 
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias
 e_jit_compiled_convolution_op*
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l_random_generator* 
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
30
41
<2
=3
K4
L5
T6
U7
c8
d9
s10
t11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45*
�
30
41
<2
=3
K4
L5
T6
U7
c8
d9
s10
t11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
)_default_save_signature
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
* 

30
41*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeAa/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeAa/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

<0
=1*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeAb/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeAb/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

K0
L1*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeBa/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeBa/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

T0
U1*

T0
U1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeBb/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeBb/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

c0
d1*

c0
d1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeCa/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeCa/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

s0
t1*

s0
t1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeCb/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeCb/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeDa/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeDa/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeDb/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeDb/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeEa/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeEa/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEencodeEb/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEencodeEb/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEtransconvE/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEtransconvE/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdecodeCa/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdecodeCa/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdecodeCb/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdecodeCb/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEtransconvC/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEtransconvC/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdecodeBa/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdecodeBa/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdecodeBb/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdecodeBb/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEtransconvB/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEtransconvB/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdecodeAa/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdecodeAa/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdecodeAb/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdecodeAb/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEtransconvA/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEtransconvA/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconvOuta/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconvOuta/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconvOutb/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconvOutb/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEPredictionMask/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEPredictionMask/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33*
<
�0
�1
�2
�3
�4
�5
�6*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89
�90
�91
�92*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/encodeAa/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/encodeAa/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/encodeAa/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/encodeAa/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/encodeAb/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/encodeAb/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/encodeAb/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/encodeAb/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/encodeBa/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeBa/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeBa/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeBa/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/encodeBb/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeBb/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeBb/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeBb/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/encodeCa/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeCa/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeCa/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeCa/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/encodeCb/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeCb/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeCb/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeCb/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/encodeDa/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeDa/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeDa/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeDa/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/encodeDb/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeDb/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeDb/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeDb/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/encodeEa/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeEa/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeEa/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeEa/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/encodeEb/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/encodeEb/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/encodeEb/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/encodeEb/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/transconvE/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/transconvE/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/transconvE/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/transconvE/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/decodeCa/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/decodeCa/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/decodeCa/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/decodeCa/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/decodeCb/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/decodeCb/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/decodeCb/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/decodeCb/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/transconvC/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/transconvC/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/transconvC/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/transconvC/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/decodeBa/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/decodeBa/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/decodeBa/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/decodeBa/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/decodeBb/kernel2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/decodeBb/kernel2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/decodeBb/bias2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/decodeBb/bias2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/transconvB/kernel2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/transconvB/kernel2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/transconvB/bias2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/transconvB/bias2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/decodeAa/kernel2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/decodeAa/kernel2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/decodeAa/bias2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/decodeAa/bias2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/decodeAb/kernel2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/decodeAb/kernel2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/decodeAb/bias2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/decodeAb/bias2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/transconvA/kernel2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/transconvA/kernel2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/transconvA/bias2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/transconvA/bias2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/convOuta/kernel2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/convOuta/kernel2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/convOuta/bias2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/convOuta/bias2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/convOutb/kernel2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/convOutb/kernel2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/convOutb/bias2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/convOutb/bias2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/PredictionMask/kernel2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/PredictionMask/kernel2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/PredictionMask/bias2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/PredictionMask/bias2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_64keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_64keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_54keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_54keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameencodeAa/kernelencodeAa/biasencodeAb/kernelencodeAb/biasencodeBa/kernelencodeBa/biasencodeBb/kernelencodeBb/biasencodeCa/kernelencodeCa/biasencodeCb/kernelencodeCb/biasencodeDa/kernelencodeDa/biasencodeDb/kernelencodeDb/biasencodeEa/kernelencodeEa/biasencodeEb/kernelencodeEb/biastransconvE/kerneltransconvE/biasdecodeCa/kerneldecodeCa/biasdecodeCb/kerneldecodeCb/biastransconvC/kerneltransconvC/biasdecodeBa/kerneldecodeBa/biasdecodeBb/kerneldecodeBb/biastransconvB/kerneltransconvB/biasdecodeAa/kerneldecodeAa/biasdecodeAb/kerneldecodeAb/biastransconvA/kerneltransconvA/biasconvOuta/kernelconvOuta/biasconvOutb/kernelconvOutb/biasPredictionMask/kernelPredictionMask/bias	iterationlearning_rateAdam/m/encodeAa/kernelAdam/v/encodeAa/kernelAdam/m/encodeAa/biasAdam/v/encodeAa/biasAdam/m/encodeAb/kernelAdam/v/encodeAb/kernelAdam/m/encodeAb/biasAdam/v/encodeAb/biasAdam/m/encodeBa/kernelAdam/v/encodeBa/kernelAdam/m/encodeBa/biasAdam/v/encodeBa/biasAdam/m/encodeBb/kernelAdam/v/encodeBb/kernelAdam/m/encodeBb/biasAdam/v/encodeBb/biasAdam/m/encodeCa/kernelAdam/v/encodeCa/kernelAdam/m/encodeCa/biasAdam/v/encodeCa/biasAdam/m/encodeCb/kernelAdam/v/encodeCb/kernelAdam/m/encodeCb/biasAdam/v/encodeCb/biasAdam/m/encodeDa/kernelAdam/v/encodeDa/kernelAdam/m/encodeDa/biasAdam/v/encodeDa/biasAdam/m/encodeDb/kernelAdam/v/encodeDb/kernelAdam/m/encodeDb/biasAdam/v/encodeDb/biasAdam/m/encodeEa/kernelAdam/v/encodeEa/kernelAdam/m/encodeEa/biasAdam/v/encodeEa/biasAdam/m/encodeEb/kernelAdam/v/encodeEb/kernelAdam/m/encodeEb/biasAdam/v/encodeEb/biasAdam/m/transconvE/kernelAdam/v/transconvE/kernelAdam/m/transconvE/biasAdam/v/transconvE/biasAdam/m/decodeCa/kernelAdam/v/decodeCa/kernelAdam/m/decodeCa/biasAdam/v/decodeCa/biasAdam/m/decodeCb/kernelAdam/v/decodeCb/kernelAdam/m/decodeCb/biasAdam/v/decodeCb/biasAdam/m/transconvC/kernelAdam/v/transconvC/kernelAdam/m/transconvC/biasAdam/v/transconvC/biasAdam/m/decodeBa/kernelAdam/v/decodeBa/kernelAdam/m/decodeBa/biasAdam/v/decodeBa/biasAdam/m/decodeBb/kernelAdam/v/decodeBb/kernelAdam/m/decodeBb/biasAdam/v/decodeBb/biasAdam/m/transconvB/kernelAdam/v/transconvB/kernelAdam/m/transconvB/biasAdam/v/transconvB/biasAdam/m/decodeAa/kernelAdam/v/decodeAa/kernelAdam/m/decodeAa/biasAdam/v/decodeAa/biasAdam/m/decodeAb/kernelAdam/v/decodeAb/kernelAdam/m/decodeAb/biasAdam/v/decodeAb/biasAdam/m/transconvA/kernelAdam/v/transconvA/kernelAdam/m/transconvA/biasAdam/v/transconvA/biasAdam/m/convOuta/kernelAdam/v/convOuta/kernelAdam/m/convOuta/biasAdam/v/convOuta/biasAdam/m/convOutb/kernelAdam/v/convOutb/kernelAdam/m/convOutb/biasAdam/v/convOutb/biasAdam/m/PredictionMask/kernelAdam/v/PredictionMask/kernelAdam/m/PredictionMask/biasAdam/v/PredictionMask/biastotal_6count_6total_5count_5total_4count_4total_3count_3total_2count_2total_1count_1totalcountConst*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_193645
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencodeAa/kernelencodeAa/biasencodeAb/kernelencodeAb/biasencodeBa/kernelencodeBa/biasencodeBb/kernelencodeBb/biasencodeCa/kernelencodeCa/biasencodeCb/kernelencodeCb/biasencodeDa/kernelencodeDa/biasencodeDb/kernelencodeDb/biasencodeEa/kernelencodeEa/biasencodeEb/kernelencodeEb/biastransconvE/kerneltransconvE/biasdecodeCa/kerneldecodeCa/biasdecodeCb/kerneldecodeCb/biastransconvC/kerneltransconvC/biasdecodeBa/kerneldecodeBa/biasdecodeBb/kerneldecodeBb/biastransconvB/kerneltransconvB/biasdecodeAa/kerneldecodeAa/biasdecodeAb/kerneldecodeAb/biastransconvA/kerneltransconvA/biasconvOuta/kernelconvOuta/biasconvOutb/kernelconvOutb/biasPredictionMask/kernelPredictionMask/bias	iterationlearning_rateAdam/m/encodeAa/kernelAdam/v/encodeAa/kernelAdam/m/encodeAa/biasAdam/v/encodeAa/biasAdam/m/encodeAb/kernelAdam/v/encodeAb/kernelAdam/m/encodeAb/biasAdam/v/encodeAb/biasAdam/m/encodeBa/kernelAdam/v/encodeBa/kernelAdam/m/encodeBa/biasAdam/v/encodeBa/biasAdam/m/encodeBb/kernelAdam/v/encodeBb/kernelAdam/m/encodeBb/biasAdam/v/encodeBb/biasAdam/m/encodeCa/kernelAdam/v/encodeCa/kernelAdam/m/encodeCa/biasAdam/v/encodeCa/biasAdam/m/encodeCb/kernelAdam/v/encodeCb/kernelAdam/m/encodeCb/biasAdam/v/encodeCb/biasAdam/m/encodeDa/kernelAdam/v/encodeDa/kernelAdam/m/encodeDa/biasAdam/v/encodeDa/biasAdam/m/encodeDb/kernelAdam/v/encodeDb/kernelAdam/m/encodeDb/biasAdam/v/encodeDb/biasAdam/m/encodeEa/kernelAdam/v/encodeEa/kernelAdam/m/encodeEa/biasAdam/v/encodeEa/biasAdam/m/encodeEb/kernelAdam/v/encodeEb/kernelAdam/m/encodeEb/biasAdam/v/encodeEb/biasAdam/m/transconvE/kernelAdam/v/transconvE/kernelAdam/m/transconvE/biasAdam/v/transconvE/biasAdam/m/decodeCa/kernelAdam/v/decodeCa/kernelAdam/m/decodeCa/biasAdam/v/decodeCa/biasAdam/m/decodeCb/kernelAdam/v/decodeCb/kernelAdam/m/decodeCb/biasAdam/v/decodeCb/biasAdam/m/transconvC/kernelAdam/v/transconvC/kernelAdam/m/transconvC/biasAdam/v/transconvC/biasAdam/m/decodeBa/kernelAdam/v/decodeBa/kernelAdam/m/decodeBa/biasAdam/v/decodeBa/biasAdam/m/decodeBb/kernelAdam/v/decodeBb/kernelAdam/m/decodeBb/biasAdam/v/decodeBb/biasAdam/m/transconvB/kernelAdam/v/transconvB/kernelAdam/m/transconvB/biasAdam/v/transconvB/biasAdam/m/decodeAa/kernelAdam/v/decodeAa/kernelAdam/m/decodeAa/biasAdam/v/decodeAa/biasAdam/m/decodeAb/kernelAdam/v/decodeAb/kernelAdam/m/decodeAb/biasAdam/v/decodeAb/biasAdam/m/transconvA/kernelAdam/v/transconvA/kernelAdam/m/transconvA/biasAdam/v/transconvA/biasAdam/m/convOuta/kernelAdam/v/convOuta/kernelAdam/m/convOuta/biasAdam/v/convOuta/biasAdam/m/convOutb/kernelAdam/v/convOutb/kernelAdam/m/convOutb/biasAdam/v/convOutb/biasAdam/m/PredictionMask/kernelAdam/v/PredictionMask/kernelAdam/m/PredictionMask/biasAdam/v/PredictionMask/biastotal_6count_6total_5count_5total_4count_4total_3count_3total_2count_2total_1count_1totalcount*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_194116ɮ
�
B
&__inference_poolC_layer_call_fn_192166

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolC_layer_call_and_return_conditional_losses_190778�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�&
�
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191688
mrimages!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@#
	unknown_9:@@

unknown_10:@%

unknown_11:@�

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�%

unknown_25:@�

unknown_26:@%

unknown_27:�@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31: @

unknown_32: $

unknown_33:@ 

unknown_34: $

unknown_35:  

unknown_36: $

unknown_37: 

unknown_38:$

unknown_39: 

unknown_40:$

unknown_41:

unknown_42:$

unknown_43:

unknown_44:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmrimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191494y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes{
y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages:&"
 
_user_specified_name191594:&"
 
_user_specified_name191596:&"
 
_user_specified_name191598:&"
 
_user_specified_name191600:&"
 
_user_specified_name191602:&"
 
_user_specified_name191604:&"
 
_user_specified_name191606:&"
 
_user_specified_name191608:&	"
 
_user_specified_name191610:&
"
 
_user_specified_name191612:&"
 
_user_specified_name191614:&"
 
_user_specified_name191616:&"
 
_user_specified_name191618:&"
 
_user_specified_name191620:&"
 
_user_specified_name191622:&"
 
_user_specified_name191624:&"
 
_user_specified_name191626:&"
 
_user_specified_name191628:&"
 
_user_specified_name191630:&"
 
_user_specified_name191632:&"
 
_user_specified_name191634:&"
 
_user_specified_name191636:&"
 
_user_specified_name191638:&"
 
_user_specified_name191640:&"
 
_user_specified_name191642:&"
 
_user_specified_name191644:&"
 
_user_specified_name191646:&"
 
_user_specified_name191648:&"
 
_user_specified_name191650:&"
 
_user_specified_name191652:&"
 
_user_specified_name191654:& "
 
_user_specified_name191656:&!"
 
_user_specified_name191658:&""
 
_user_specified_name191660:&#"
 
_user_specified_name191662:&$"
 
_user_specified_name191664:&%"
 
_user_specified_name191666:&&"
 
_user_specified_name191668:&'"
 
_user_specified_name191670:&("
 
_user_specified_name191672:&)"
 
_user_specified_name191674:&*"
 
_user_specified_name191676:&+"
 
_user_specified_name191678:&,"
 
_user_specified_name191680:&-"
 
_user_specified_name191682:&."
 
_user_specified_name191684
�
�
/__inference_PredictionMask_layer_call_fn_192688

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_PredictionMask_layer_call_and_return_conditional_losses_191358y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192682:&"
 
_user_specified_name192684
�
�
)__inference_decodeAa_layer_call_fn_192553

inputs!
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeAa_layer_call_and_return_conditional_losses_191281w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs:&"
 
_user_specified_name192547:&"
 
_user_specified_name192549
�!
�
F__inference_transconvA_layer_call_and_return_conditional_losses_190990

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
B
&__inference_poolA_layer_call_fn_192028

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolA_layer_call_and_return_conditional_losses_190720�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
T
(__inference_concatB_layer_call_fn_192537
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatB_layer_call_and_return_conditional_losses_191269h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@ :���������@@ :Y U
/
_output_shapes
:���������@@ 
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������@@ 
"
_user_specified_name
inputs_1
�
�
D__inference_encodeEb_layer_call_and_return_conditional_losses_191162

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
k
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192141

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
T
(__inference_concatC_layer_call_fn_192442
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatC_layer_call_and_return_conditional_losses_191224i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������  @:���������  @:Y U
/
_output_shapes
:���������  @
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������  @
"
_user_specified_name
inputs_1
�
n
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_190806

inputs
identity�I
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*J
_output_shapes8
6:4�������������������������������������
IdentityIdentitydropout/SelectV2:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_decodeBb_layer_call_and_return_conditional_losses_192489

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_decodeAb_layer_call_fn_192573

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeAb_layer_call_and_return_conditional_losses_191297w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs:&"
 
_user_specified_name192567:&"
 
_user_specified_name192569
�
�
D__inference_decodeBa_layer_call_and_return_conditional_losses_191236

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�!
�
F__inference_transconvB_layer_call_and_return_conditional_losses_192531

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_convOutb_layer_call_and_return_conditional_losses_191342

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
m
4__inference_spatial_dropout2d_1_layer_call_fn_192196

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_190806�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*J
_output_shapes8
6:4������������������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
]
A__inference_poolA_layer_call_and_return_conditional_losses_192033

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
ʚ
�4
!__inference__wrapped_model_190715
mrimagesW
=dunet_brats_decathlon_encodeaa_conv2d_readvariableop_resource:L
>dunet_brats_decathlon_encodeaa_biasadd_readvariableop_resource:W
=dunet_brats_decathlon_encodeab_conv2d_readvariableop_resource:L
>dunet_brats_decathlon_encodeab_biasadd_readvariableop_resource:W
=dunet_brats_decathlon_encodeba_conv2d_readvariableop_resource: L
>dunet_brats_decathlon_encodeba_biasadd_readvariableop_resource: W
=dunet_brats_decathlon_encodebb_conv2d_readvariableop_resource:  L
>dunet_brats_decathlon_encodebb_biasadd_readvariableop_resource: W
=dunet_brats_decathlon_encodeca_conv2d_readvariableop_resource: @L
>dunet_brats_decathlon_encodeca_biasadd_readvariableop_resource:@W
=dunet_brats_decathlon_encodecb_conv2d_readvariableop_resource:@@L
>dunet_brats_decathlon_encodecb_biasadd_readvariableop_resource:@X
=dunet_brats_decathlon_encodeda_conv2d_readvariableop_resource:@�M
>dunet_brats_decathlon_encodeda_biasadd_readvariableop_resource:	�Y
=dunet_brats_decathlon_encodedb_conv2d_readvariableop_resource:��M
>dunet_brats_decathlon_encodedb_biasadd_readvariableop_resource:	�Y
=dunet_brats_decathlon_encodeea_conv2d_readvariableop_resource:��M
>dunet_brats_decathlon_encodeea_biasadd_readvariableop_resource:	�Y
=dunet_brats_decathlon_encodeeb_conv2d_readvariableop_resource:��M
>dunet_brats_decathlon_encodeeb_biasadd_readvariableop_resource:	�e
Idunet_brats_decathlon_transconve_conv2d_transpose_readvariableop_resource:��O
@dunet_brats_decathlon_transconve_biasadd_readvariableop_resource:	�Y
=dunet_brats_decathlon_decodeca_conv2d_readvariableop_resource:��M
>dunet_brats_decathlon_decodeca_biasadd_readvariableop_resource:	�Y
=dunet_brats_decathlon_decodecb_conv2d_readvariableop_resource:��M
>dunet_brats_decathlon_decodecb_biasadd_readvariableop_resource:	�d
Idunet_brats_decathlon_transconvc_conv2d_transpose_readvariableop_resource:@�N
@dunet_brats_decathlon_transconvc_biasadd_readvariableop_resource:@X
=dunet_brats_decathlon_decodeba_conv2d_readvariableop_resource:�@L
>dunet_brats_decathlon_decodeba_biasadd_readvariableop_resource:@W
=dunet_brats_decathlon_decodebb_conv2d_readvariableop_resource:@@L
>dunet_brats_decathlon_decodebb_biasadd_readvariableop_resource:@c
Idunet_brats_decathlon_transconvb_conv2d_transpose_readvariableop_resource: @N
@dunet_brats_decathlon_transconvb_biasadd_readvariableop_resource: W
=dunet_brats_decathlon_decodeaa_conv2d_readvariableop_resource:@ L
>dunet_brats_decathlon_decodeaa_biasadd_readvariableop_resource: W
=dunet_brats_decathlon_decodeab_conv2d_readvariableop_resource:  L
>dunet_brats_decathlon_decodeab_biasadd_readvariableop_resource: c
Idunet_brats_decathlon_transconva_conv2d_transpose_readvariableop_resource: N
@dunet_brats_decathlon_transconva_biasadd_readvariableop_resource:W
=dunet_brats_decathlon_convouta_conv2d_readvariableop_resource: L
>dunet_brats_decathlon_convouta_biasadd_readvariableop_resource:W
=dunet_brats_decathlon_convoutb_conv2d_readvariableop_resource:L
>dunet_brats_decathlon_convoutb_biasadd_readvariableop_resource:]
Cdunet_brats_decathlon_predictionmask_conv2d_readvariableop_resource:R
Ddunet_brats_decathlon_predictionmask_biasadd_readvariableop_resource:
identity��<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp�;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp�62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp�52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp�82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp�82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp�82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp�82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp�A2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp�
52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeaa_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
&2DUNet_Brats_Decathlon/encodeAa/Conv2DConv2Dmrimages=2DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeaa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
'2DUNet_Brats_Decathlon/encodeAa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeAa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
$2DUNet_Brats_Decathlon/encodeAa/ReluRelu02DUNet_Brats_Decathlon/encodeAa/BiasAdd:output:0*
T0*1
_output_shapes
:������������
52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeab_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
&2DUNet_Brats_Decathlon/encodeAb/Conv2DConv2D22DUNet_Brats_Decathlon/encodeAa/Relu:activations:0=2DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeab_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
'2DUNet_Brats_Decathlon/encodeAb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeAb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
$2DUNet_Brats_Decathlon/encodeAb/ReluRelu02DUNet_Brats_Decathlon/encodeAb/BiasAdd:output:0*
T0*1
_output_shapes
:������������
$2DUNet_Brats_Decathlon/poolA/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeAb/Relu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingVALID*
strides
�
52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeba_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&2DUNet_Brats_Decathlon/encodeBa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolA/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeba_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
'2DUNet_Brats_Decathlon/encodeBa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeBa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ �
$2DUNet_Brats_Decathlon/encodeBa/ReluRelu02DUNet_Brats_Decathlon/encodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ �
52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodebb_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
&2DUNet_Brats_Decathlon/encodeBb/Conv2DConv2D22DUNet_Brats_Decathlon/encodeBa/Relu:activations:0=2DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodebb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
'2DUNet_Brats_Decathlon/encodeBb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeBb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ �
$2DUNet_Brats_Decathlon/encodeBb/ReluRelu02DUNet_Brats_Decathlon/encodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ �
$2DUNet_Brats_Decathlon/poolB/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeBb/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingVALID*
strides
�
52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeca_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&2DUNet_Brats_Decathlon/encodeCa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolB/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeca_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'2DUNet_Brats_Decathlon/encodeCa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeCa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
$2DUNet_Brats_Decathlon/encodeCa/ReluRelu02DUNet_Brats_Decathlon/encodeCa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @�
12DUNet_Brats_Decathlon/spatial_dropout2d/IdentityIdentity22DUNet_Brats_Decathlon/encodeCa/Relu:activations:0*
T0*/
_output_shapes
:���������  @�
52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodecb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
&2DUNet_Brats_Decathlon/encodeCb/Conv2DConv2D:2DUNet_Brats_Decathlon/spatial_dropout2d/Identity:output:0=2DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodecb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'2DUNet_Brats_Decathlon/encodeCb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeCb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
$2DUNet_Brats_Decathlon/encodeCb/ReluRelu02DUNet_Brats_Decathlon/encodeCb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @�
$2DUNet_Brats_Decathlon/poolC/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeCb/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeda_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&2DUNet_Brats_Decathlon/encodeDa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolC/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeda_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'2DUNet_Brats_Decathlon/encodeDa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeDa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
$2DUNet_Brats_Decathlon/encodeDa/ReluRelu02DUNet_Brats_Decathlon/encodeDa/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
32DUNet_Brats_Decathlon/spatial_dropout2d_1/IdentityIdentity22DUNet_Brats_Decathlon/encodeDa/Relu:activations:0*
T0*0
_output_shapes
:�����������
52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodedb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&2DUNet_Brats_Decathlon/encodeDb/Conv2DConv2D<2DUNet_Brats_Decathlon/spatial_dropout2d_1/Identity:output:0=2DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodedb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'2DUNet_Brats_Decathlon/encodeDb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeDb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
$2DUNet_Brats_Decathlon/encodeDb/ReluRelu02DUNet_Brats_Decathlon/encodeDb/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
$2DUNet_Brats_Decathlon/poolD/MaxPoolMaxPool22DUNet_Brats_Decathlon/encodeDb/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeea_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&2DUNet_Brats_Decathlon/encodeEa/Conv2DConv2D-2DUNet_Brats_Decathlon/poolD/MaxPool:output:0=2DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeea_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'2DUNet_Brats_Decathlon/encodeEa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeEa/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
$2DUNet_Brats_Decathlon/encodeEa/ReluRelu02DUNet_Brats_Decathlon/encodeEa/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_encodeeb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&2DUNet_Brats_Decathlon/encodeEb/Conv2DConv2D22DUNet_Brats_Decathlon/encodeEa/Relu:activations:0=2DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_encodeeb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'2DUNet_Brats_Decathlon/encodeEb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/encodeEb/Conv2D:output:0>2DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
$2DUNet_Brats_Decathlon/encodeEb/ReluRelu02DUNet_Brats_Decathlon/encodeEb/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
'2DUNet_Brats_Decathlon/transconvE/ShapeShape22DUNet_Brats_Decathlon/encodeEb/Relu:activations:0*
T0*
_output_shapes
::��
52DUNet_Brats_Decathlon/transconvE/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
72DUNet_Brats_Decathlon/transconvE/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
72DUNet_Brats_Decathlon/transconvE/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/2DUNet_Brats_Decathlon/transconvE/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvE/Shape:output:0>2DUNet_Brats_Decathlon/transconvE/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvE/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvE/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)2DUNet_Brats_Decathlon/transconvE/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)2DUNet_Brats_Decathlon/transconvE/stack/2Const*
_output_shapes
: *
dtype0*
value	B :l
)2DUNet_Brats_Decathlon/transconvE/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
'2DUNet_Brats_Decathlon/transconvE/stackPack82DUNet_Brats_Decathlon/transconvE/strided_slice:output:022DUNet_Brats_Decathlon/transconvE/stack/1:output:022DUNet_Brats_Decathlon/transconvE/stack/2:output:022DUNet_Brats_Decathlon/transconvE/stack/3:output:0*
N*
T0*
_output_shapes
:�
72DUNet_Brats_Decathlon/transconvE/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
92DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
92DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
12DUNet_Brats_Decathlon/transconvE/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvE/stack:output:0@2DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvE/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
A2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconve_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
22DUNet_Brats_Decathlon/transconvE/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvE/stack:output:0I2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/encodeEb/Relu:activations:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconve_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)2DUNet_Brats_Decathlon/transconvE/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvE/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������l
*2DUNet_Brats_Decathlon/concatD/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
%2DUNet_Brats_Decathlon/concatD/concatConcatV222DUNet_Brats_Decathlon/transconvE/BiasAdd:output:022DUNet_Brats_Decathlon/encodeDb/Relu:activations:032DUNet_Brats_Decathlon/concatD/concat/axis:output:0*
N*
T0*0
_output_shapes
:�����������
52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeca_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&2DUNet_Brats_Decathlon/decodeCa/Conv2DConv2D.2DUNet_Brats_Decathlon/concatD/concat:output:0=2DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeca_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'2DUNet_Brats_Decathlon/decodeCa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeCa/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
$2DUNet_Brats_Decathlon/decodeCa/ReluRelu02DUNet_Brats_Decathlon/decodeCa/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodecb_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&2DUNet_Brats_Decathlon/decodeCb/Conv2DConv2D22DUNet_Brats_Decathlon/decodeCa/Relu:activations:0=2DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodecb_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'2DUNet_Brats_Decathlon/decodeCb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeCb/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
$2DUNet_Brats_Decathlon/decodeCb/ReluRelu02DUNet_Brats_Decathlon/decodeCb/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
'2DUNet_Brats_Decathlon/transconvC/ShapeShape22DUNet_Brats_Decathlon/decodeCb/Relu:activations:0*
T0*
_output_shapes
::��
52DUNet_Brats_Decathlon/transconvC/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
72DUNet_Brats_Decathlon/transconvC/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
72DUNet_Brats_Decathlon/transconvC/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/2DUNet_Brats_Decathlon/transconvC/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvC/Shape:output:0>2DUNet_Brats_Decathlon/transconvC/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvC/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvC/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)2DUNet_Brats_Decathlon/transconvC/stack/1Const*
_output_shapes
: *
dtype0*
value	B : k
)2DUNet_Brats_Decathlon/transconvC/stack/2Const*
_output_shapes
: *
dtype0*
value	B : k
)2DUNet_Brats_Decathlon/transconvC/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
'2DUNet_Brats_Decathlon/transconvC/stackPack82DUNet_Brats_Decathlon/transconvC/strided_slice:output:022DUNet_Brats_Decathlon/transconvC/stack/1:output:022DUNet_Brats_Decathlon/transconvC/stack/2:output:022DUNet_Brats_Decathlon/transconvC/stack/3:output:0*
N*
T0*
_output_shapes
:�
72DUNet_Brats_Decathlon/transconvC/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
92DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
92DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
12DUNet_Brats_Decathlon/transconvC/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvC/stack:output:0@2DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvC/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
A2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconvc_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
22DUNet_Brats_Decathlon/transconvC/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvC/stack:output:0I2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/decodeCb/Relu:activations:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconvc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
)2DUNet_Brats_Decathlon/transconvC/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvC/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @l
*2DUNet_Brats_Decathlon/concatC/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
%2DUNet_Brats_Decathlon/concatC/concatConcatV222DUNet_Brats_Decathlon/transconvC/BiasAdd:output:022DUNet_Brats_Decathlon/encodeCb/Relu:activations:032DUNet_Brats_Decathlon/concatC/concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  ��
52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeba_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
&2DUNet_Brats_Decathlon/decodeBa/Conv2DConv2D.2DUNet_Brats_Decathlon/concatC/concat:output:0=2DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeba_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'2DUNet_Brats_Decathlon/decodeBa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeBa/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
$2DUNet_Brats_Decathlon/decodeBa/ReluRelu02DUNet_Brats_Decathlon/decodeBa/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @�
52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodebb_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
&2DUNet_Brats_Decathlon/decodeBb/Conv2DConv2D22DUNet_Brats_Decathlon/decodeBa/Relu:activations:0=2DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodebb_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'2DUNet_Brats_Decathlon/decodeBb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeBb/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
$2DUNet_Brats_Decathlon/decodeBb/ReluRelu02DUNet_Brats_Decathlon/decodeBb/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @�
'2DUNet_Brats_Decathlon/transconvB/ShapeShape22DUNet_Brats_Decathlon/decodeBb/Relu:activations:0*
T0*
_output_shapes
::��
52DUNet_Brats_Decathlon/transconvB/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
72DUNet_Brats_Decathlon/transconvB/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
72DUNet_Brats_Decathlon/transconvB/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/2DUNet_Brats_Decathlon/transconvB/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvB/Shape:output:0>2DUNet_Brats_Decathlon/transconvB/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvB/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvB/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)2DUNet_Brats_Decathlon/transconvB/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@k
)2DUNet_Brats_Decathlon/transconvB/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@k
)2DUNet_Brats_Decathlon/transconvB/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
'2DUNet_Brats_Decathlon/transconvB/stackPack82DUNet_Brats_Decathlon/transconvB/strided_slice:output:022DUNet_Brats_Decathlon/transconvB/stack/1:output:022DUNet_Brats_Decathlon/transconvB/stack/2:output:022DUNet_Brats_Decathlon/transconvB/stack/3:output:0*
N*
T0*
_output_shapes
:�
72DUNet_Brats_Decathlon/transconvB/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
92DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
92DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
12DUNet_Brats_Decathlon/transconvB/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvB/stack:output:0@2DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvB/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
A2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconvb_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
22DUNet_Brats_Decathlon/transconvB/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvB/stack:output:0I2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/decodeBb/Relu:activations:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
�
82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconvb_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)2DUNet_Brats_Decathlon/transconvB/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvB/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ l
*2DUNet_Brats_Decathlon/concatB/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
%2DUNet_Brats_Decathlon/concatB/concatConcatV222DUNet_Brats_Decathlon/transconvB/BiasAdd:output:022DUNet_Brats_Decathlon/encodeBb/Relu:activations:032DUNet_Brats_Decathlon/concatB/concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@�
52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeaa_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
&2DUNet_Brats_Decathlon/decodeAa/Conv2DConv2D.2DUNet_Brats_Decathlon/concatB/concat:output:0=2DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeaa_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
'2DUNet_Brats_Decathlon/decodeAa/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeAa/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ �
$2DUNet_Brats_Decathlon/decodeAa/ReluRelu02DUNet_Brats_Decathlon/decodeAa/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ �
52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_decodeab_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
&2DUNet_Brats_Decathlon/decodeAb/Conv2DConv2D22DUNet_Brats_Decathlon/decodeAa/Relu:activations:0=2DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_decodeab_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
'2DUNet_Brats_Decathlon/decodeAb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/decodeAb/Conv2D:output:0>2DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ �
$2DUNet_Brats_Decathlon/decodeAb/ReluRelu02DUNet_Brats_Decathlon/decodeAb/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@ �
'2DUNet_Brats_Decathlon/transconvA/ShapeShape22DUNet_Brats_Decathlon/decodeAb/Relu:activations:0*
T0*
_output_shapes
::��
52DUNet_Brats_Decathlon/transconvA/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
72DUNet_Brats_Decathlon/transconvA/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
72DUNet_Brats_Decathlon/transconvA/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/2DUNet_Brats_Decathlon/transconvA/strided_sliceStridedSlice02DUNet_Brats_Decathlon/transconvA/Shape:output:0>2DUNet_Brats_Decathlon/transconvA/strided_slice/stack:output:0@2DUNet_Brats_Decathlon/transconvA/strided_slice/stack_1:output:0@2DUNet_Brats_Decathlon/transconvA/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
)2DUNet_Brats_Decathlon/transconvA/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�l
)2DUNet_Brats_Decathlon/transconvA/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�k
)2DUNet_Brats_Decathlon/transconvA/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
'2DUNet_Brats_Decathlon/transconvA/stackPack82DUNet_Brats_Decathlon/transconvA/strided_slice:output:022DUNet_Brats_Decathlon/transconvA/stack/1:output:022DUNet_Brats_Decathlon/transconvA/stack/2:output:022DUNet_Brats_Decathlon/transconvA/stack/3:output:0*
N*
T0*
_output_shapes
:�
72DUNet_Brats_Decathlon/transconvA/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
92DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
92DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
12DUNet_Brats_Decathlon/transconvA/strided_slice_1StridedSlice02DUNet_Brats_Decathlon/transconvA/stack:output:0@2DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack:output:0B2DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_1:output:0B2DUNet_Brats_Decathlon/transconvA/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
A2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOpReadVariableOpIdunet_brats_decathlon_transconva_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
22DUNet_Brats_Decathlon/transconvA/conv2d_transposeConv2DBackpropInput02DUNet_Brats_Decathlon/transconvA/stack:output:0I2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp:value:022DUNet_Brats_Decathlon/decodeAb/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOpReadVariableOp@dunet_brats_decathlon_transconva_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)2DUNet_Brats_Decathlon/transconvA/BiasAddBiasAdd;2DUNet_Brats_Decathlon/transconvA/conv2d_transpose:output:0@2DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������l
*2DUNet_Brats_Decathlon/concatA/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
%2DUNet_Brats_Decathlon/concatA/concatConcatV222DUNet_Brats_Decathlon/transconvA/BiasAdd:output:022DUNet_Brats_Decathlon/encodeAb/Relu:activations:032DUNet_Brats_Decathlon/concatA/concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� �
52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_convouta_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&2DUNet_Brats_Decathlon/convOuta/Conv2DConv2D.2DUNet_Brats_Decathlon/concatA/concat:output:0=2DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_convouta_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
'2DUNet_Brats_Decathlon/convOuta/BiasAddBiasAdd/2DUNet_Brats_Decathlon/convOuta/Conv2D:output:0>2DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
$2DUNet_Brats_Decathlon/convOuta/ReluRelu02DUNet_Brats_Decathlon/convOuta/BiasAdd:output:0*
T0*1
_output_shapes
:������������
52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOpReadVariableOp=dunet_brats_decathlon_convoutb_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
&2DUNet_Brats_Decathlon/convOutb/Conv2DConv2D22DUNet_Brats_Decathlon/convOuta/Relu:activations:0=2DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOpReadVariableOp>dunet_brats_decathlon_convoutb_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
'2DUNet_Brats_Decathlon/convOutb/BiasAddBiasAdd/2DUNet_Brats_Decathlon/convOutb/Conv2D:output:0>2DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
$2DUNet_Brats_Decathlon/convOutb/ReluRelu02DUNet_Brats_Decathlon/convOutb/BiasAdd:output:0*
T0*1
_output_shapes
:������������
;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOpReadVariableOpCdunet_brats_decathlon_predictionmask_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,2DUNet_Brats_Decathlon/PredictionMask/Conv2DConv2D22DUNet_Brats_Decathlon/convOutb/Relu:activations:0C2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
�
<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOpReadVariableOpDdunet_brats_decathlon_predictionmask_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-2DUNet_Brats_Decathlon/PredictionMask/BiasAddBiasAdd52DUNet_Brats_Decathlon/PredictionMask/Conv2D:output:0D2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
-2DUNet_Brats_Decathlon/PredictionMask/SigmoidSigmoid62DUNet_Brats_Decathlon/PredictionMask/BiasAdd:output:0*
T0*1
_output_shapes
:������������
IdentityIdentity12DUNet_Brats_Decathlon/PredictionMask/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp=^2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp<^2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp7^2DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp6^2DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp9^2DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOpB^2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes{
y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp<2DUNet_Brats_Decathlon/PredictionMask/BiasAdd/ReadVariableOp2z
;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp;2DUNet_Brats_Decathlon/PredictionMask/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/convOuta/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/convOuta/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/convOutb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/convOutb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeAa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeAa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeAb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeAb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeBa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeBa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeBb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeBb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeCa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeCa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/decodeCb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/decodeCb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeAa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeAa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeAb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeAb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeBa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeBa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeBb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeBb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeCa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeCa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeCb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeCb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeDa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeDa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeDb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeDb/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeEa/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeEa/Conv2D/ReadVariableOp2p
62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp62DUNet_Brats_Decathlon/encodeEb/BiasAdd/ReadVariableOp2n
52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp52DUNet_Brats_Decathlon/encodeEb/Conv2D/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvA/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvA/conv2d_transpose/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvB/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvB/conv2d_transpose/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvC/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvC/conv2d_transpose/ReadVariableOp2t
82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp82DUNet_Brats_Decathlon/transconvE/BiasAdd/ReadVariableOp2�
A2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOpA2DUNet_Brats_Decathlon/transconvE/conv2d_transpose/ReadVariableOp:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource
�
�
)__inference_encodeBa_layer_call_fn_192042

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeBa_layer_call_and_return_conditional_losses_191045w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:&"
 
_user_specified_name192036:&"
 
_user_specified_name192038
�
�
+__inference_transconvA_layer_call_fn_192593

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvA_layer_call_and_return_conditional_losses_190990�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name192587:&"
 
_user_specified_name192589
�
�
)__inference_convOuta_layer_call_fn_192648

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_convOuta_layer_call_and_return_conditional_losses_191326y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name192642:&"
 
_user_specified_name192644
�
�
D__inference_decodeCb_layer_call_and_return_conditional_losses_192394

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
m
C__inference_concatA_layer_call_and_return_conditional_losses_191314

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:YU
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�!
�
F__inference_transconvE_layer_call_and_return_conditional_losses_192341

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeEb_layer_call_and_return_conditional_losses_192299

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeCa_layer_call_and_return_conditional_losses_191078

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_encodeEb_layer_call_fn_192288

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeEb_layer_call_and_return_conditional_losses_191162x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192282:&"
 
_user_specified_name192284
�!
�
F__inference_transconvC_layer_call_and_return_conditional_losses_192436

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeDb_layer_call_and_return_conditional_losses_192249

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�!
�
F__inference_transconvA_layer_call_and_return_conditional_losses_192626

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
]
A__inference_poolA_layer_call_and_return_conditional_losses_190720

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
T
(__inference_concatD_layer_call_fn_192347
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatD_layer_call_and_return_conditional_losses_191179i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
�
)__inference_decodeCb_layer_call_fn_192383

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeCb_layer_call_and_return_conditional_losses_191207x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192377:&"
 
_user_specified_name192379
�
�
D__inference_decodeBb_layer_call_and_return_conditional_losses_191252

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
o
C__inference_concatC_layer_call_and_return_conditional_losses_192449
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������  @:���������  @:Y U
/
_output_shapes
:���������  @
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������  @
"
_user_specified_name
inputs_1
�
�
D__inference_encodeEa_layer_call_and_return_conditional_losses_192279

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191494
mrimages)
encodeaa_191368:
encodeaa_191370:)
encodeab_191373:
encodeab_191375:)
encodeba_191379: 
encodeba_191381: )
encodebb_191384:  
encodebb_191386: )
encodeca_191390: @
encodeca_191392:@)
encodecb_191396:@@
encodecb_191398:@*
encodeda_191402:@�
encodeda_191404:	�+
encodedb_191408:��
encodedb_191410:	�+
encodeea_191414:��
encodeea_191416:	�+
encodeeb_191419:��
encodeeb_191421:	�-
transconve_191424:�� 
transconve_191426:	�+
decodeca_191430:��
decodeca_191432:	�+
decodecb_191435:��
decodecb_191437:	�,
transconvc_191440:@�
transconvc_191442:@*
decodeba_191446:�@
decodeba_191448:@)
decodebb_191451:@@
decodebb_191453:@+
transconvb_191456: @
transconvb_191458: )
decodeaa_191462:@ 
decodeaa_191464: )
decodeab_191467:  
decodeab_191469: +
transconva_191472: 
transconva_191474:)
convouta_191478: 
convouta_191480:)
convoutb_191483:
convoutb_191485:/
predictionmask_191488:#
predictionmask_191490:
identity��&PredictionMask/StatefulPartitionedCall� convOuta/StatefulPartitionedCall� convOutb/StatefulPartitionedCall� decodeAa/StatefulPartitionedCall� decodeAb/StatefulPartitionedCall� decodeBa/StatefulPartitionedCall� decodeBb/StatefulPartitionedCall� decodeCa/StatefulPartitionedCall� decodeCb/StatefulPartitionedCall� encodeAa/StatefulPartitionedCall� encodeAb/StatefulPartitionedCall� encodeBa/StatefulPartitionedCall� encodeBb/StatefulPartitionedCall� encodeCa/StatefulPartitionedCall� encodeCb/StatefulPartitionedCall� encodeDa/StatefulPartitionedCall� encodeDb/StatefulPartitionedCall� encodeEa/StatefulPartitionedCall� encodeEb/StatefulPartitionedCall�"transconvA/StatefulPartitionedCall�"transconvB/StatefulPartitionedCall�"transconvC/StatefulPartitionedCall�"transconvE/StatefulPartitionedCall�
 encodeAa/StatefulPartitionedCallStatefulPartitionedCallmrimagesencodeaa_191368encodeaa_191370*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeAa_layer_call_and_return_conditional_losses_191012�
 encodeAb/StatefulPartitionedCallStatefulPartitionedCall)encodeAa/StatefulPartitionedCall:output:0encodeab_191373encodeab_191375*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeAb_layer_call_and_return_conditional_losses_191028�
poolA/PartitionedCallPartitionedCall)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolA_layer_call_and_return_conditional_losses_190720�
 encodeBa/StatefulPartitionedCallStatefulPartitionedCallpoolA/PartitionedCall:output:0encodeba_191379encodeba_191381*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeBa_layer_call_and_return_conditional_losses_191045�
 encodeBb/StatefulPartitionedCallStatefulPartitionedCall)encodeBa/StatefulPartitionedCall:output:0encodebb_191384encodebb_191386*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeBb_layer_call_and_return_conditional_losses_191061�
poolB/PartitionedCallPartitionedCall)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolB_layer_call_and_return_conditional_losses_190730�
 encodeCa/StatefulPartitionedCallStatefulPartitionedCallpoolB/PartitionedCall:output:0encodeca_191390encodeca_191392*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeCa_layer_call_and_return_conditional_losses_191078�
!spatial_dropout2d/PartitionedCallPartitionedCall)encodeCa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_190763�
 encodeCb/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout2d/PartitionedCall:output:0encodecb_191396encodecb_191398*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeCb_layer_call_and_return_conditional_losses_191095�
poolC/PartitionedCallPartitionedCall)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolC_layer_call_and_return_conditional_losses_190778�
 encodeDa/StatefulPartitionedCallStatefulPartitionedCallpoolC/PartitionedCall:output:0encodeda_191402encodeda_191404*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeDa_layer_call_and_return_conditional_losses_191112�
#spatial_dropout2d_1/PartitionedCallPartitionedCall)encodeDa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_190811�
 encodeDb/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout2d_1/PartitionedCall:output:0encodedb_191408encodedb_191410*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeDb_layer_call_and_return_conditional_losses_191129�
poolD/PartitionedCallPartitionedCall)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolD_layer_call_and_return_conditional_losses_190826�
 encodeEa/StatefulPartitionedCallStatefulPartitionedCallpoolD/PartitionedCall:output:0encodeea_191414encodeea_191416*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeEa_layer_call_and_return_conditional_losses_191146�
 encodeEb/StatefulPartitionedCallStatefulPartitionedCall)encodeEa/StatefulPartitionedCall:output:0encodeeb_191419encodeeb_191421*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeEb_layer_call_and_return_conditional_losses_191162�
"transconvE/StatefulPartitionedCallStatefulPartitionedCall)encodeEb/StatefulPartitionedCall:output:0transconve_191424transconve_191426*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvE_layer_call_and_return_conditional_losses_190864�
concatD/PartitionedCallPartitionedCall+transconvE/StatefulPartitionedCall:output:0)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatD_layer_call_and_return_conditional_losses_191179�
 decodeCa/StatefulPartitionedCallStatefulPartitionedCall concatD/PartitionedCall:output:0decodeca_191430decodeca_191432*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeCa_layer_call_and_return_conditional_losses_191191�
 decodeCb/StatefulPartitionedCallStatefulPartitionedCall)decodeCa/StatefulPartitionedCall:output:0decodecb_191435decodecb_191437*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeCb_layer_call_and_return_conditional_losses_191207�
"transconvC/StatefulPartitionedCallStatefulPartitionedCall)decodeCb/StatefulPartitionedCall:output:0transconvc_191440transconvc_191442*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvC_layer_call_and_return_conditional_losses_190906�
concatC/PartitionedCallPartitionedCall+transconvC/StatefulPartitionedCall:output:0)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatC_layer_call_and_return_conditional_losses_191224�
 decodeBa/StatefulPartitionedCallStatefulPartitionedCall concatC/PartitionedCall:output:0decodeba_191446decodeba_191448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeBa_layer_call_and_return_conditional_losses_191236�
 decodeBb/StatefulPartitionedCallStatefulPartitionedCall)decodeBa/StatefulPartitionedCall:output:0decodebb_191451decodebb_191453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeBb_layer_call_and_return_conditional_losses_191252�
"transconvB/StatefulPartitionedCallStatefulPartitionedCall)decodeBb/StatefulPartitionedCall:output:0transconvb_191456transconvb_191458*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvB_layer_call_and_return_conditional_losses_190948�
concatB/PartitionedCallPartitionedCall+transconvB/StatefulPartitionedCall:output:0)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatB_layer_call_and_return_conditional_losses_191269�
 decodeAa/StatefulPartitionedCallStatefulPartitionedCall concatB/PartitionedCall:output:0decodeaa_191462decodeaa_191464*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeAa_layer_call_and_return_conditional_losses_191281�
 decodeAb/StatefulPartitionedCallStatefulPartitionedCall)decodeAa/StatefulPartitionedCall:output:0decodeab_191467decodeab_191469*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeAb_layer_call_and_return_conditional_losses_191297�
"transconvA/StatefulPartitionedCallStatefulPartitionedCall)decodeAb/StatefulPartitionedCall:output:0transconva_191472transconva_191474*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvA_layer_call_and_return_conditional_losses_190990�
concatA/PartitionedCallPartitionedCall+transconvA/StatefulPartitionedCall:output:0)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatA_layer_call_and_return_conditional_losses_191314�
 convOuta/StatefulPartitionedCallStatefulPartitionedCall concatA/PartitionedCall:output:0convouta_191478convouta_191480*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_convOuta_layer_call_and_return_conditional_losses_191326�
 convOutb/StatefulPartitionedCallStatefulPartitionedCall)convOuta/StatefulPartitionedCall:output:0convoutb_191483convoutb_191485*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_convOutb_layer_call_and_return_conditional_losses_191342�
&PredictionMask/StatefulPartitionedCallStatefulPartitionedCall)convOutb/StatefulPartitionedCall:output:0predictionmask_191488predictionmask_191490*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_PredictionMask_layer_call_and_return_conditional_losses_191358�
IdentityIdentity/PredictionMask/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp'^PredictionMask/StatefulPartitionedCall!^convOuta/StatefulPartitionedCall!^convOutb/StatefulPartitionedCall!^decodeAa/StatefulPartitionedCall!^decodeAb/StatefulPartitionedCall!^decodeBa/StatefulPartitionedCall!^decodeBb/StatefulPartitionedCall!^decodeCa/StatefulPartitionedCall!^decodeCb/StatefulPartitionedCall!^encodeAa/StatefulPartitionedCall!^encodeAb/StatefulPartitionedCall!^encodeBa/StatefulPartitionedCall!^encodeBb/StatefulPartitionedCall!^encodeCa/StatefulPartitionedCall!^encodeCb/StatefulPartitionedCall!^encodeDa/StatefulPartitionedCall!^encodeDb/StatefulPartitionedCall!^encodeEa/StatefulPartitionedCall!^encodeEb/StatefulPartitionedCall#^transconvA/StatefulPartitionedCall#^transconvB/StatefulPartitionedCall#^transconvC/StatefulPartitionedCall#^transconvE/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes{
y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&PredictionMask/StatefulPartitionedCall&PredictionMask/StatefulPartitionedCall2D
 convOuta/StatefulPartitionedCall convOuta/StatefulPartitionedCall2D
 convOutb/StatefulPartitionedCall convOutb/StatefulPartitionedCall2D
 decodeAa/StatefulPartitionedCall decodeAa/StatefulPartitionedCall2D
 decodeAb/StatefulPartitionedCall decodeAb/StatefulPartitionedCall2D
 decodeBa/StatefulPartitionedCall decodeBa/StatefulPartitionedCall2D
 decodeBb/StatefulPartitionedCall decodeBb/StatefulPartitionedCall2D
 decodeCa/StatefulPartitionedCall decodeCa/StatefulPartitionedCall2D
 decodeCb/StatefulPartitionedCall decodeCb/StatefulPartitionedCall2D
 encodeAa/StatefulPartitionedCall encodeAa/StatefulPartitionedCall2D
 encodeAb/StatefulPartitionedCall encodeAb/StatefulPartitionedCall2D
 encodeBa/StatefulPartitionedCall encodeBa/StatefulPartitionedCall2D
 encodeBb/StatefulPartitionedCall encodeBb/StatefulPartitionedCall2D
 encodeCa/StatefulPartitionedCall encodeCa/StatefulPartitionedCall2D
 encodeCb/StatefulPartitionedCall encodeCb/StatefulPartitionedCall2D
 encodeDa/StatefulPartitionedCall encodeDa/StatefulPartitionedCall2D
 encodeDb/StatefulPartitionedCall encodeDb/StatefulPartitionedCall2D
 encodeEa/StatefulPartitionedCall encodeEa/StatefulPartitionedCall2D
 encodeEb/StatefulPartitionedCall encodeEb/StatefulPartitionedCall2H
"transconvA/StatefulPartitionedCall"transconvA/StatefulPartitionedCall2H
"transconvB/StatefulPartitionedCall"transconvB/StatefulPartitionedCall2H
"transconvC/StatefulPartitionedCall"transconvC/StatefulPartitionedCall2H
"transconvE/StatefulPartitionedCall"transconvE/StatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages:&"
 
_user_specified_name191368:&"
 
_user_specified_name191370:&"
 
_user_specified_name191373:&"
 
_user_specified_name191375:&"
 
_user_specified_name191379:&"
 
_user_specified_name191381:&"
 
_user_specified_name191384:&"
 
_user_specified_name191386:&	"
 
_user_specified_name191390:&
"
 
_user_specified_name191392:&"
 
_user_specified_name191396:&"
 
_user_specified_name191398:&"
 
_user_specified_name191402:&"
 
_user_specified_name191404:&"
 
_user_specified_name191408:&"
 
_user_specified_name191410:&"
 
_user_specified_name191414:&"
 
_user_specified_name191416:&"
 
_user_specified_name191419:&"
 
_user_specified_name191421:&"
 
_user_specified_name191424:&"
 
_user_specified_name191426:&"
 
_user_specified_name191430:&"
 
_user_specified_name191432:&"
 
_user_specified_name191435:&"
 
_user_specified_name191437:&"
 
_user_specified_name191440:&"
 
_user_specified_name191442:&"
 
_user_specified_name191446:&"
 
_user_specified_name191448:&"
 
_user_specified_name191451:& "
 
_user_specified_name191453:&!"
 
_user_specified_name191456:&""
 
_user_specified_name191458:&#"
 
_user_specified_name191462:&$"
 
_user_specified_name191464:&%"
 
_user_specified_name191467:&&"
 
_user_specified_name191469:&'"
 
_user_specified_name191472:&("
 
_user_specified_name191474:&)"
 
_user_specified_name191478:&*"
 
_user_specified_name191480:&+"
 
_user_specified_name191483:&,"
 
_user_specified_name191485:&-"
 
_user_specified_name191488:&."
 
_user_specified_name191490
�
�
D__inference_decodeBa_layer_call_and_return_conditional_losses_192469

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
l
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192136

inputs
identity�I
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*J
_output_shapes8
6:4�������������������������������������
IdentityIdentitydropout/SelectV2:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_decodeCb_layer_call_and_return_conditional_losses_191207

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
o
C__inference_concatA_layer_call_and_return_conditional_losses_192639
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:����������� a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs_0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs_1
�
�
+__inference_transconvE_layer_call_fn_192308

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvE_layer_call_and_return_conditional_losses_190864�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:&"
 
_user_specified_name192302:&"
 
_user_specified_name192304
�
n
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192224

inputs
identity�I
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*J
_output_shapes8
6:4�������������������������������������
IdentityIdentitydropout/SelectV2:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
]
A__inference_poolC_layer_call_and_return_conditional_losses_190778

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_encodeCb_layer_call_and_return_conditional_losses_191095

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
B
&__inference_poolB_layer_call_fn_192078

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolB_layer_call_and_return_conditional_losses_190730�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_encodeAa_layer_call_and_return_conditional_losses_192003

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_decodeBa_layer_call_fn_192458

inputs"
unknown:�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeBa_layer_call_and_return_conditional_losses_191236w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs:&"
 
_user_specified_name192452:&"
 
_user_specified_name192454
�&
�
$__inference_signature_wrapper_191983
mrimages!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@#
	unknown_9:@@

unknown_10:@%

unknown_11:@�

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�%

unknown_25:@�

unknown_26:@%

unknown_27:�@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31: @

unknown_32: $

unknown_33:@ 

unknown_34: $

unknown_35:  

unknown_36: $

unknown_37: 

unknown_38:$

unknown_39: 

unknown_40:$

unknown_41:

unknown_42:$

unknown_43:

unknown_44:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmrimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_190715y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes{
y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages:&"
 
_user_specified_name191889:&"
 
_user_specified_name191891:&"
 
_user_specified_name191893:&"
 
_user_specified_name191895:&"
 
_user_specified_name191897:&"
 
_user_specified_name191899:&"
 
_user_specified_name191901:&"
 
_user_specified_name191903:&	"
 
_user_specified_name191905:&
"
 
_user_specified_name191907:&"
 
_user_specified_name191909:&"
 
_user_specified_name191911:&"
 
_user_specified_name191913:&"
 
_user_specified_name191915:&"
 
_user_specified_name191917:&"
 
_user_specified_name191919:&"
 
_user_specified_name191921:&"
 
_user_specified_name191923:&"
 
_user_specified_name191925:&"
 
_user_specified_name191927:&"
 
_user_specified_name191929:&"
 
_user_specified_name191931:&"
 
_user_specified_name191933:&"
 
_user_specified_name191935:&"
 
_user_specified_name191937:&"
 
_user_specified_name191939:&"
 
_user_specified_name191941:&"
 
_user_specified_name191943:&"
 
_user_specified_name191945:&"
 
_user_specified_name191947:&"
 
_user_specified_name191949:& "
 
_user_specified_name191951:&!"
 
_user_specified_name191953:&""
 
_user_specified_name191955:&#"
 
_user_specified_name191957:&$"
 
_user_specified_name191959:&%"
 
_user_specified_name191961:&&"
 
_user_specified_name191963:&'"
 
_user_specified_name191965:&("
 
_user_specified_name191967:&)"
 
_user_specified_name191969:&*"
 
_user_specified_name191971:&+"
 
_user_specified_name191973:&,"
 
_user_specified_name191975:&-"
 
_user_specified_name191977:&."
 
_user_specified_name191979
�
m
C__inference_concatD_layer_call_and_return_conditional_losses_191179

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:XT
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_encodeEa_layer_call_fn_192268

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeEa_layer_call_and_return_conditional_losses_191146x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192262:&"
 
_user_specified_name192264
�!
�
F__inference_transconvC_layer_call_and_return_conditional_losses_190906

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_decodeCa_layer_call_and_return_conditional_losses_192374

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
]
A__inference_poolB_layer_call_and_return_conditional_losses_190730

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_encodeBa_layer_call_and_return_conditional_losses_191045

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
o
C__inference_concatB_layer_call_and_return_conditional_losses_192544
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@ :���������@@ :Y U
/
_output_shapes
:���������@@ 
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������@@ 
"
_user_specified_name
inputs_1
�
T
(__inference_concatA_layer_call_fn_192632
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatA_layer_call_and_return_conditional_losses_191314j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::�����������:�����������:[ W
1
_output_shapes
:�����������
"
_user_specified_name
inputs_0:[W
1
_output_shapes
:�����������
"
_user_specified_name
inputs_1
�
B
&__inference_poolD_layer_call_fn_192254

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolD_layer_call_and_return_conditional_losses_190826�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_encodeBb_layer_call_fn_192062

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeBb_layer_call_and_return_conditional_losses_191061w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs:&"
 
_user_specified_name192056:&"
 
_user_specified_name192058
�
�
D__inference_encodeDa_layer_call_and_return_conditional_losses_192191

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_decodeBb_layer_call_fn_192478

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeBb_layer_call_and_return_conditional_losses_191252w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs:&"
 
_user_specified_name192472:&"
 
_user_specified_name192474
�
m
C__inference_concatC_layer_call_and_return_conditional_losses_191224

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:���������  �`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������  @:���������  @:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
m
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_190811

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_encodeDa_layer_call_fn_192180

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeDa_layer_call_and_return_conditional_losses_191112x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:&"
 
_user_specified_name192174:&"
 
_user_specified_name192176
�
�
D__inference_encodeAb_layer_call_and_return_conditional_losses_191028

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_PredictionMask_layer_call_and_return_conditional_losses_191358

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_decodeAb_layer_call_and_return_conditional_losses_192584

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeAb_layer_call_and_return_conditional_losses_192023

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
]
A__inference_poolD_layer_call_and_return_conditional_losses_190826

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_convOuta_layer_call_and_return_conditional_losses_191326

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
m
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192229

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
k
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_190763

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4������������������������������������~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4������������������������������������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_decodeAa_layer_call_and_return_conditional_losses_191281

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeDa_layer_call_and_return_conditional_losses_191112

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_decodeCa_layer_call_fn_192363

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeCa_layer_call_and_return_conditional_losses_191191x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192357:&"
 
_user_specified_name192359
�
]
A__inference_poolB_layer_call_and_return_conditional_losses_192083

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
o
C__inference_concatD_layer_call_and_return_conditional_losses_192354
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1
҃	
��
__inference__traced_save_193645
file_prefix@
&read_disablecopyonread_encodeaa_kernel:4
&read_1_disablecopyonread_encodeaa_bias:B
(read_2_disablecopyonread_encodeab_kernel:4
&read_3_disablecopyonread_encodeab_bias:B
(read_4_disablecopyonread_encodeba_kernel: 4
&read_5_disablecopyonread_encodeba_bias: B
(read_6_disablecopyonread_encodebb_kernel:  4
&read_7_disablecopyonread_encodebb_bias: B
(read_8_disablecopyonread_encodeca_kernel: @4
&read_9_disablecopyonread_encodeca_bias:@C
)read_10_disablecopyonread_encodecb_kernel:@@5
'read_11_disablecopyonread_encodecb_bias:@D
)read_12_disablecopyonread_encodeda_kernel:@�6
'read_13_disablecopyonread_encodeda_bias:	�E
)read_14_disablecopyonread_encodedb_kernel:��6
'read_15_disablecopyonread_encodedb_bias:	�E
)read_16_disablecopyonread_encodeea_kernel:��6
'read_17_disablecopyonread_encodeea_bias:	�E
)read_18_disablecopyonread_encodeeb_kernel:��6
'read_19_disablecopyonread_encodeeb_bias:	�G
+read_20_disablecopyonread_transconve_kernel:��8
)read_21_disablecopyonread_transconve_bias:	�E
)read_22_disablecopyonread_decodeca_kernel:��6
'read_23_disablecopyonread_decodeca_bias:	�E
)read_24_disablecopyonread_decodecb_kernel:��6
'read_25_disablecopyonread_decodecb_bias:	�F
+read_26_disablecopyonread_transconvc_kernel:@�7
)read_27_disablecopyonread_transconvc_bias:@D
)read_28_disablecopyonread_decodeba_kernel:�@5
'read_29_disablecopyonread_decodeba_bias:@C
)read_30_disablecopyonread_decodebb_kernel:@@5
'read_31_disablecopyonread_decodebb_bias:@E
+read_32_disablecopyonread_transconvb_kernel: @7
)read_33_disablecopyonread_transconvb_bias: C
)read_34_disablecopyonread_decodeaa_kernel:@ 5
'read_35_disablecopyonread_decodeaa_bias: C
)read_36_disablecopyonread_decodeab_kernel:  5
'read_37_disablecopyonread_decodeab_bias: E
+read_38_disablecopyonread_transconva_kernel: 7
)read_39_disablecopyonread_transconva_bias:C
)read_40_disablecopyonread_convouta_kernel: 5
'read_41_disablecopyonread_convouta_bias:C
)read_42_disablecopyonread_convoutb_kernel:5
'read_43_disablecopyonread_convoutb_bias:I
/read_44_disablecopyonread_predictionmask_kernel:;
-read_45_disablecopyonread_predictionmask_bias:-
#read_46_disablecopyonread_iteration:	 1
'read_47_disablecopyonread_learning_rate: J
0read_48_disablecopyonread_adam_m_encodeaa_kernel:J
0read_49_disablecopyonread_adam_v_encodeaa_kernel:<
.read_50_disablecopyonread_adam_m_encodeaa_bias:<
.read_51_disablecopyonread_adam_v_encodeaa_bias:J
0read_52_disablecopyonread_adam_m_encodeab_kernel:J
0read_53_disablecopyonread_adam_v_encodeab_kernel:<
.read_54_disablecopyonread_adam_m_encodeab_bias:<
.read_55_disablecopyonread_adam_v_encodeab_bias:J
0read_56_disablecopyonread_adam_m_encodeba_kernel: J
0read_57_disablecopyonread_adam_v_encodeba_kernel: <
.read_58_disablecopyonread_adam_m_encodeba_bias: <
.read_59_disablecopyonread_adam_v_encodeba_bias: J
0read_60_disablecopyonread_adam_m_encodebb_kernel:  J
0read_61_disablecopyonread_adam_v_encodebb_kernel:  <
.read_62_disablecopyonread_adam_m_encodebb_bias: <
.read_63_disablecopyonread_adam_v_encodebb_bias: J
0read_64_disablecopyonread_adam_m_encodeca_kernel: @J
0read_65_disablecopyonread_adam_v_encodeca_kernel: @<
.read_66_disablecopyonread_adam_m_encodeca_bias:@<
.read_67_disablecopyonread_adam_v_encodeca_bias:@J
0read_68_disablecopyonread_adam_m_encodecb_kernel:@@J
0read_69_disablecopyonread_adam_v_encodecb_kernel:@@<
.read_70_disablecopyonread_adam_m_encodecb_bias:@<
.read_71_disablecopyonread_adam_v_encodecb_bias:@K
0read_72_disablecopyonread_adam_m_encodeda_kernel:@�K
0read_73_disablecopyonread_adam_v_encodeda_kernel:@�=
.read_74_disablecopyonread_adam_m_encodeda_bias:	�=
.read_75_disablecopyonread_adam_v_encodeda_bias:	�L
0read_76_disablecopyonread_adam_m_encodedb_kernel:��L
0read_77_disablecopyonread_adam_v_encodedb_kernel:��=
.read_78_disablecopyonread_adam_m_encodedb_bias:	�=
.read_79_disablecopyonread_adam_v_encodedb_bias:	�L
0read_80_disablecopyonread_adam_m_encodeea_kernel:��L
0read_81_disablecopyonread_adam_v_encodeea_kernel:��=
.read_82_disablecopyonread_adam_m_encodeea_bias:	�=
.read_83_disablecopyonread_adam_v_encodeea_bias:	�L
0read_84_disablecopyonread_adam_m_encodeeb_kernel:��L
0read_85_disablecopyonread_adam_v_encodeeb_kernel:��=
.read_86_disablecopyonread_adam_m_encodeeb_bias:	�=
.read_87_disablecopyonread_adam_v_encodeeb_bias:	�N
2read_88_disablecopyonread_adam_m_transconve_kernel:��N
2read_89_disablecopyonread_adam_v_transconve_kernel:��?
0read_90_disablecopyonread_adam_m_transconve_bias:	�?
0read_91_disablecopyonread_adam_v_transconve_bias:	�L
0read_92_disablecopyonread_adam_m_decodeca_kernel:��L
0read_93_disablecopyonread_adam_v_decodeca_kernel:��=
.read_94_disablecopyonread_adam_m_decodeca_bias:	�=
.read_95_disablecopyonread_adam_v_decodeca_bias:	�L
0read_96_disablecopyonread_adam_m_decodecb_kernel:��L
0read_97_disablecopyonread_adam_v_decodecb_kernel:��=
.read_98_disablecopyonread_adam_m_decodecb_bias:	�=
.read_99_disablecopyonread_adam_v_decodecb_bias:	�N
3read_100_disablecopyonread_adam_m_transconvc_kernel:@�N
3read_101_disablecopyonread_adam_v_transconvc_kernel:@�?
1read_102_disablecopyonread_adam_m_transconvc_bias:@?
1read_103_disablecopyonread_adam_v_transconvc_bias:@L
1read_104_disablecopyonread_adam_m_decodeba_kernel:�@L
1read_105_disablecopyonread_adam_v_decodeba_kernel:�@=
/read_106_disablecopyonread_adam_m_decodeba_bias:@=
/read_107_disablecopyonread_adam_v_decodeba_bias:@K
1read_108_disablecopyonread_adam_m_decodebb_kernel:@@K
1read_109_disablecopyonread_adam_v_decodebb_kernel:@@=
/read_110_disablecopyonread_adam_m_decodebb_bias:@=
/read_111_disablecopyonread_adam_v_decodebb_bias:@M
3read_112_disablecopyonread_adam_m_transconvb_kernel: @M
3read_113_disablecopyonread_adam_v_transconvb_kernel: @?
1read_114_disablecopyonread_adam_m_transconvb_bias: ?
1read_115_disablecopyonread_adam_v_transconvb_bias: K
1read_116_disablecopyonread_adam_m_decodeaa_kernel:@ K
1read_117_disablecopyonread_adam_v_decodeaa_kernel:@ =
/read_118_disablecopyonread_adam_m_decodeaa_bias: =
/read_119_disablecopyonread_adam_v_decodeaa_bias: K
1read_120_disablecopyonread_adam_m_decodeab_kernel:  K
1read_121_disablecopyonread_adam_v_decodeab_kernel:  =
/read_122_disablecopyonread_adam_m_decodeab_bias: =
/read_123_disablecopyonread_adam_v_decodeab_bias: M
3read_124_disablecopyonread_adam_m_transconva_kernel: M
3read_125_disablecopyonread_adam_v_transconva_kernel: ?
1read_126_disablecopyonread_adam_m_transconva_bias:?
1read_127_disablecopyonread_adam_v_transconva_bias:K
1read_128_disablecopyonread_adam_m_convouta_kernel: K
1read_129_disablecopyonread_adam_v_convouta_kernel: =
/read_130_disablecopyonread_adam_m_convouta_bias:=
/read_131_disablecopyonread_adam_v_convouta_bias:K
1read_132_disablecopyonread_adam_m_convoutb_kernel:K
1read_133_disablecopyonread_adam_v_convoutb_kernel:=
/read_134_disablecopyonread_adam_m_convoutb_bias:=
/read_135_disablecopyonread_adam_v_convoutb_bias:Q
7read_136_disablecopyonread_adam_m_predictionmask_kernel:Q
7read_137_disablecopyonread_adam_v_predictionmask_kernel:C
5read_138_disablecopyonread_adam_m_predictionmask_bias:C
5read_139_disablecopyonread_adam_v_predictionmask_bias:,
"read_140_disablecopyonread_total_6: ,
"read_141_disablecopyonread_count_6: ,
"read_142_disablecopyonread_total_5: ,
"read_143_disablecopyonread_count_5: ,
"read_144_disablecopyonread_total_4: ,
"read_145_disablecopyonread_count_4: ,
"read_146_disablecopyonread_total_3: ,
"read_147_disablecopyonread_count_3: ,
"read_148_disablecopyonread_total_2: ,
"read_149_disablecopyonread_count_2: ,
"read_150_disablecopyonread_total_1: ,
"read_151_disablecopyonread_count_1: *
 read_152_disablecopyonread_total: *
 read_153_disablecopyonread_count: 
savev2_const
identity_309��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_100/DisableCopyOnRead�Read_100/ReadVariableOp�Read_101/DisableCopyOnRead�Read_101/ReadVariableOp�Read_102/DisableCopyOnRead�Read_102/ReadVariableOp�Read_103/DisableCopyOnRead�Read_103/ReadVariableOp�Read_104/DisableCopyOnRead�Read_104/ReadVariableOp�Read_105/DisableCopyOnRead�Read_105/ReadVariableOp�Read_106/DisableCopyOnRead�Read_106/ReadVariableOp�Read_107/DisableCopyOnRead�Read_107/ReadVariableOp�Read_108/DisableCopyOnRead�Read_108/ReadVariableOp�Read_109/DisableCopyOnRead�Read_109/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_110/DisableCopyOnRead�Read_110/ReadVariableOp�Read_111/DisableCopyOnRead�Read_111/ReadVariableOp�Read_112/DisableCopyOnRead�Read_112/ReadVariableOp�Read_113/DisableCopyOnRead�Read_113/ReadVariableOp�Read_114/DisableCopyOnRead�Read_114/ReadVariableOp�Read_115/DisableCopyOnRead�Read_115/ReadVariableOp�Read_116/DisableCopyOnRead�Read_116/ReadVariableOp�Read_117/DisableCopyOnRead�Read_117/ReadVariableOp�Read_118/DisableCopyOnRead�Read_118/ReadVariableOp�Read_119/DisableCopyOnRead�Read_119/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_120/DisableCopyOnRead�Read_120/ReadVariableOp�Read_121/DisableCopyOnRead�Read_121/ReadVariableOp�Read_122/DisableCopyOnRead�Read_122/ReadVariableOp�Read_123/DisableCopyOnRead�Read_123/ReadVariableOp�Read_124/DisableCopyOnRead�Read_124/ReadVariableOp�Read_125/DisableCopyOnRead�Read_125/ReadVariableOp�Read_126/DisableCopyOnRead�Read_126/ReadVariableOp�Read_127/DisableCopyOnRead�Read_127/ReadVariableOp�Read_128/DisableCopyOnRead�Read_128/ReadVariableOp�Read_129/DisableCopyOnRead�Read_129/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_130/DisableCopyOnRead�Read_130/ReadVariableOp�Read_131/DisableCopyOnRead�Read_131/ReadVariableOp�Read_132/DisableCopyOnRead�Read_132/ReadVariableOp�Read_133/DisableCopyOnRead�Read_133/ReadVariableOp�Read_134/DisableCopyOnRead�Read_134/ReadVariableOp�Read_135/DisableCopyOnRead�Read_135/ReadVariableOp�Read_136/DisableCopyOnRead�Read_136/ReadVariableOp�Read_137/DisableCopyOnRead�Read_137/ReadVariableOp�Read_138/DisableCopyOnRead�Read_138/ReadVariableOp�Read_139/DisableCopyOnRead�Read_139/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_140/DisableCopyOnRead�Read_140/ReadVariableOp�Read_141/DisableCopyOnRead�Read_141/ReadVariableOp�Read_142/DisableCopyOnRead�Read_142/ReadVariableOp�Read_143/DisableCopyOnRead�Read_143/ReadVariableOp�Read_144/DisableCopyOnRead�Read_144/ReadVariableOp�Read_145/DisableCopyOnRead�Read_145/ReadVariableOp�Read_146/DisableCopyOnRead�Read_146/ReadVariableOp�Read_147/DisableCopyOnRead�Read_147/ReadVariableOp�Read_148/DisableCopyOnRead�Read_148/ReadVariableOp�Read_149/DisableCopyOnRead�Read_149/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_150/DisableCopyOnRead�Read_150/ReadVariableOp�Read_151/DisableCopyOnRead�Read_151/ReadVariableOp�Read_152/DisableCopyOnRead�Read_152/ReadVariableOp�Read_153/DisableCopyOnRead�Read_153/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_encodeaa_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_encodeaa_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_encodeaa_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_encodeaa_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_encodeab_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_encodeab_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_encodeab_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_encodeab_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_encodeba_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_encodeba_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_encodeba_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_encodeba_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_encodebb_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_encodebb_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:  z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_encodebb_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_encodebb_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_encodeca_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_encodeca_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
: @z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_encodeca_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_encodeca_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_encodecb_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_encodecb_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_encodecb_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_encodecb_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_encodeda_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_encodeda_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_encodeda_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_encodeda_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_encodedb_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_encodedb_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*(
_output_shapes
:��|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_encodedb_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_encodedb_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_encodeea_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_encodeea_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*(
_output_shapes
:��|
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_encodeea_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_encodeea_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_encodeeb_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_encodeeb_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*(
_output_shapes
:��|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_encodeeb_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_encodeeb_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead+read_20_disablecopyonread_transconve_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp+read_20_disablecopyonread_transconve_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*(
_output_shapes
:��~
Read_21/DisableCopyOnReadDisableCopyOnRead)read_21_disablecopyonread_transconve_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp)read_21_disablecopyonread_transconve_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_22/DisableCopyOnReadDisableCopyOnRead)read_22_disablecopyonread_decodeca_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp)read_22_disablecopyonread_decodeca_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*(
_output_shapes
:��|
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_decodeca_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_decodeca_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_decodecb_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_decodecb_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*(
_output_shapes
:��|
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_decodecb_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_decodecb_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead+read_26_disablecopyonread_transconvc_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp+read_26_disablecopyonread_transconvc_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�~
Read_27/DisableCopyOnReadDisableCopyOnRead)read_27_disablecopyonread_transconvc_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp)read_27_disablecopyonread_transconvc_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_28/DisableCopyOnReadDisableCopyOnRead)read_28_disablecopyonread_decodeba_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp)read_28_disablecopyonread_decodeba_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0x
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@n
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@|
Read_29/DisableCopyOnReadDisableCopyOnRead'read_29_disablecopyonread_decodeba_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp'read_29_disablecopyonread_decodeba_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_30/DisableCopyOnReadDisableCopyOnRead)read_30_disablecopyonread_decodebb_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp)read_30_disablecopyonread_decodebb_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@|
Read_31/DisableCopyOnReadDisableCopyOnRead'read_31_disablecopyonread_decodebb_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp'read_31_disablecopyonread_decodebb_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_32/DisableCopyOnReadDisableCopyOnRead+read_32_disablecopyonread_transconvb_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp+read_32_disablecopyonread_transconvb_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
: @~
Read_33/DisableCopyOnReadDisableCopyOnRead)read_33_disablecopyonread_transconvb_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp)read_33_disablecopyonread_transconvb_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_34/DisableCopyOnReadDisableCopyOnRead)read_34_disablecopyonread_decodeaa_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp)read_34_disablecopyonread_decodeaa_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ |
Read_35/DisableCopyOnReadDisableCopyOnRead'read_35_disablecopyonread_decodeaa_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp'read_35_disablecopyonread_decodeaa_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_36/DisableCopyOnReadDisableCopyOnRead)read_36_disablecopyonread_decodeab_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp)read_36_disablecopyonread_decodeab_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:  |
Read_37/DisableCopyOnReadDisableCopyOnRead'read_37_disablecopyonread_decodeab_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp'read_37_disablecopyonread_decodeab_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_38/DisableCopyOnReadDisableCopyOnRead+read_38_disablecopyonread_transconva_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp+read_38_disablecopyonread_transconva_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*&
_output_shapes
: ~
Read_39/DisableCopyOnReadDisableCopyOnRead)read_39_disablecopyonread_transconva_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp)read_39_disablecopyonread_transconva_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_40/DisableCopyOnReadDisableCopyOnRead)read_40_disablecopyonread_convouta_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp)read_40_disablecopyonread_convouta_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
: |
Read_41/DisableCopyOnReadDisableCopyOnRead'read_41_disablecopyonread_convouta_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp'read_41_disablecopyonread_convouta_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_42/DisableCopyOnReadDisableCopyOnRead)read_42_disablecopyonread_convoutb_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp)read_42_disablecopyonread_convoutb_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_43/DisableCopyOnReadDisableCopyOnRead'read_43_disablecopyonread_convoutb_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp'read_43_disablecopyonread_convoutb_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_predictionmask_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_predictionmask_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead-read_45_disablecopyonread_predictionmask_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp-read_45_disablecopyonread_predictionmask_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_46/DisableCopyOnReadDisableCopyOnRead#read_46_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp#read_46_disablecopyonread_iteration^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_47/DisableCopyOnReadDisableCopyOnRead'read_47_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp'read_47_disablecopyonread_learning_rate^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_encodeaa_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_encodeaa_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_encodeaa_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_encodeaa_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_encodeaa_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_encodeaa_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_encodeaa_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_encodeaa_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_52/DisableCopyOnReadDisableCopyOnRead0read_52_disablecopyonread_adam_m_encodeab_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp0read_52_disablecopyonread_adam_m_encodeab_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_53/DisableCopyOnReadDisableCopyOnRead0read_53_disablecopyonread_adam_v_encodeab_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp0read_53_disablecopyonread_adam_v_encodeab_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_54/DisableCopyOnReadDisableCopyOnRead.read_54_disablecopyonread_adam_m_encodeab_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp.read_54_disablecopyonread_adam_m_encodeab_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_55/DisableCopyOnReadDisableCopyOnRead.read_55_disablecopyonread_adam_v_encodeab_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp.read_55_disablecopyonread_adam_v_encodeab_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_56/DisableCopyOnReadDisableCopyOnRead0read_56_disablecopyonread_adam_m_encodeba_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp0read_56_disablecopyonread_adam_m_encodeba_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_57/DisableCopyOnReadDisableCopyOnRead0read_57_disablecopyonread_adam_v_encodeba_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp0read_57_disablecopyonread_adam_v_encodeba_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_58/DisableCopyOnReadDisableCopyOnRead.read_58_disablecopyonread_adam_m_encodeba_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp.read_58_disablecopyonread_adam_m_encodeba_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_59/DisableCopyOnReadDisableCopyOnRead.read_59_disablecopyonread_adam_v_encodeba_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp.read_59_disablecopyonread_adam_v_encodeba_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_60/DisableCopyOnReadDisableCopyOnRead0read_60_disablecopyonread_adam_m_encodebb_kernel"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp0read_60_disablecopyonread_adam_m_encodebb_kernel^Read_60/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0x
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_61/DisableCopyOnReadDisableCopyOnRead0read_61_disablecopyonread_adam_v_encodebb_kernel"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp0read_61_disablecopyonread_adam_v_encodebb_kernel^Read_61/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0x
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_62/DisableCopyOnReadDisableCopyOnRead.read_62_disablecopyonread_adam_m_encodebb_bias"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp.read_62_disablecopyonread_adam_m_encodebb_bias^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_63/DisableCopyOnReadDisableCopyOnRead.read_63_disablecopyonread_adam_v_encodebb_bias"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp.read_63_disablecopyonread_adam_v_encodebb_bias^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_64/DisableCopyOnReadDisableCopyOnRead0read_64_disablecopyonread_adam_m_encodeca_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp0read_64_disablecopyonread_adam_m_encodeca_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0x
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_65/DisableCopyOnReadDisableCopyOnRead0read_65_disablecopyonread_adam_v_encodeca_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp0read_65_disablecopyonread_adam_v_encodeca_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0x
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_66/DisableCopyOnReadDisableCopyOnRead.read_66_disablecopyonread_adam_m_encodeca_bias"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp.read_66_disablecopyonread_adam_m_encodeca_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_67/DisableCopyOnReadDisableCopyOnRead.read_67_disablecopyonread_adam_v_encodeca_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp.read_67_disablecopyonread_adam_v_encodeca_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_68/DisableCopyOnReadDisableCopyOnRead0read_68_disablecopyonread_adam_m_encodecb_kernel"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp0read_68_disablecopyonread_adam_m_encodecb_kernel^Read_68/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_69/DisableCopyOnReadDisableCopyOnRead0read_69_disablecopyonread_adam_v_encodecb_kernel"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp0read_69_disablecopyonread_adam_v_encodecb_kernel^Read_69/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_70/DisableCopyOnReadDisableCopyOnRead.read_70_disablecopyonread_adam_m_encodecb_bias"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp.read_70_disablecopyonread_adam_m_encodecb_bias^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_71/DisableCopyOnReadDisableCopyOnRead.read_71_disablecopyonread_adam_v_encodecb_bias"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp.read_71_disablecopyonread_adam_v_encodecb_bias^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_72/DisableCopyOnReadDisableCopyOnRead0read_72_disablecopyonread_adam_m_encodeda_kernel"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp0read_72_disablecopyonread_adam_m_encodeda_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0y
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_73/DisableCopyOnReadDisableCopyOnRead0read_73_disablecopyonread_adam_v_encodeda_kernel"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp0read_73_disablecopyonread_adam_v_encodeda_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0y
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_74/DisableCopyOnReadDisableCopyOnRead.read_74_disablecopyonread_adam_m_encodeda_bias"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp.read_74_disablecopyonread_adam_m_encodeda_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_75/DisableCopyOnReadDisableCopyOnRead.read_75_disablecopyonread_adam_v_encodeda_bias"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp.read_75_disablecopyonread_adam_v_encodeda_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_76/DisableCopyOnReadDisableCopyOnRead0read_76_disablecopyonread_adam_m_encodedb_kernel"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp0read_76_disablecopyonread_adam_m_encodedb_kernel^Read_76/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_77/DisableCopyOnReadDisableCopyOnRead0read_77_disablecopyonread_adam_v_encodedb_kernel"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp0read_77_disablecopyonread_adam_v_encodedb_kernel^Read_77/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_78/DisableCopyOnReadDisableCopyOnRead.read_78_disablecopyonread_adam_m_encodedb_bias"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp.read_78_disablecopyonread_adam_m_encodedb_bias^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_79/DisableCopyOnReadDisableCopyOnRead.read_79_disablecopyonread_adam_v_encodedb_bias"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp.read_79_disablecopyonread_adam_v_encodedb_bias^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_80/DisableCopyOnReadDisableCopyOnRead0read_80_disablecopyonread_adam_m_encodeea_kernel"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp0read_80_disablecopyonread_adam_m_encodeea_kernel^Read_80/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_81/DisableCopyOnReadDisableCopyOnRead0read_81_disablecopyonread_adam_v_encodeea_kernel"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp0read_81_disablecopyonread_adam_v_encodeea_kernel^Read_81/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_82/DisableCopyOnReadDisableCopyOnRead.read_82_disablecopyonread_adam_m_encodeea_bias"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp.read_82_disablecopyonread_adam_m_encodeea_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_83/DisableCopyOnReadDisableCopyOnRead.read_83_disablecopyonread_adam_v_encodeea_bias"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp.read_83_disablecopyonread_adam_v_encodeea_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_84/DisableCopyOnReadDisableCopyOnRead0read_84_disablecopyonread_adam_m_encodeeb_kernel"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp0read_84_disablecopyonread_adam_m_encodeeb_kernel^Read_84/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_85/DisableCopyOnReadDisableCopyOnRead0read_85_disablecopyonread_adam_v_encodeeb_kernel"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp0read_85_disablecopyonread_adam_v_encodeeb_kernel^Read_85/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_86/DisableCopyOnReadDisableCopyOnRead.read_86_disablecopyonread_adam_m_encodeeb_bias"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp.read_86_disablecopyonread_adam_m_encodeeb_bias^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_87/DisableCopyOnReadDisableCopyOnRead.read_87_disablecopyonread_adam_v_encodeeb_bias"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp.read_87_disablecopyonread_adam_v_encodeeb_bias^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_88/DisableCopyOnReadDisableCopyOnRead2read_88_disablecopyonread_adam_m_transconve_kernel"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp2read_88_disablecopyonread_adam_m_transconve_kernel^Read_88/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_89/DisableCopyOnReadDisableCopyOnRead2read_89_disablecopyonread_adam_v_transconve_kernel"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp2read_89_disablecopyonread_adam_v_transconve_kernel^Read_89/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_90/DisableCopyOnReadDisableCopyOnRead0read_90_disablecopyonread_adam_m_transconve_bias"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp0read_90_disablecopyonread_adam_m_transconve_bias^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_91/DisableCopyOnReadDisableCopyOnRead0read_91_disablecopyonread_adam_v_transconve_bias"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp0read_91_disablecopyonread_adam_v_transconve_bias^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_92/DisableCopyOnReadDisableCopyOnRead0read_92_disablecopyonread_adam_m_decodeca_kernel"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp0read_92_disablecopyonread_adam_m_decodeca_kernel^Read_92/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_93/DisableCopyOnReadDisableCopyOnRead0read_93_disablecopyonread_adam_v_decodeca_kernel"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp0read_93_disablecopyonread_adam_v_decodeca_kernel^Read_93/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_94/DisableCopyOnReadDisableCopyOnRead.read_94_disablecopyonread_adam_m_decodeca_bias"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOp.read_94_disablecopyonread_adam_m_decodeca_bias^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_95/DisableCopyOnReadDisableCopyOnRead.read_95_disablecopyonread_adam_v_decodeca_bias"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOp.read_95_disablecopyonread_adam_v_decodeca_bias^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_96/DisableCopyOnReadDisableCopyOnRead0read_96_disablecopyonread_adam_m_decodecb_kernel"/device:CPU:0*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOp0read_96_disablecopyonread_adam_m_decodecb_kernel^Read_96/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_97/DisableCopyOnReadDisableCopyOnRead0read_97_disablecopyonread_adam_v_decodecb_kernel"/device:CPU:0*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOp0read_97_disablecopyonread_adam_v_decodecb_kernel^Read_97/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_98/DisableCopyOnReadDisableCopyOnRead.read_98_disablecopyonread_adam_m_decodecb_bias"/device:CPU:0*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOp.read_98_disablecopyonread_adam_m_decodecb_bias^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_99/DisableCopyOnReadDisableCopyOnRead.read_99_disablecopyonread_adam_v_decodecb_bias"/device:CPU:0*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOp.read_99_disablecopyonread_adam_v_decodecb_bias^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_100/DisableCopyOnReadDisableCopyOnRead3read_100_disablecopyonread_adam_m_transconvc_kernel"/device:CPU:0*
_output_shapes
 �
Read_100/ReadVariableOpReadVariableOp3read_100_disablecopyonread_adam_m_transconvc_kernel^Read_100/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0z
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_101/DisableCopyOnReadDisableCopyOnRead3read_101_disablecopyonread_adam_v_transconvc_kernel"/device:CPU:0*
_output_shapes
 �
Read_101/ReadVariableOpReadVariableOp3read_101_disablecopyonread_adam_v_transconvc_kernel^Read_101/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0z
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_102/DisableCopyOnReadDisableCopyOnRead1read_102_disablecopyonread_adam_m_transconvc_bias"/device:CPU:0*
_output_shapes
 �
Read_102/ReadVariableOpReadVariableOp1read_102_disablecopyonread_adam_m_transconvc_bias^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_103/DisableCopyOnReadDisableCopyOnRead1read_103_disablecopyonread_adam_v_transconvc_bias"/device:CPU:0*
_output_shapes
 �
Read_103/ReadVariableOpReadVariableOp1read_103_disablecopyonread_adam_v_transconvc_bias^Read_103/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_104/DisableCopyOnReadDisableCopyOnRead1read_104_disablecopyonread_adam_m_decodeba_kernel"/device:CPU:0*
_output_shapes
 �
Read_104/ReadVariableOpReadVariableOp1read_104_disablecopyonread_adam_m_decodeba_kernel^Read_104/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0z
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@p
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_105/DisableCopyOnReadDisableCopyOnRead1read_105_disablecopyonread_adam_v_decodeba_kernel"/device:CPU:0*
_output_shapes
 �
Read_105/ReadVariableOpReadVariableOp1read_105_disablecopyonread_adam_v_decodeba_kernel^Read_105/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0z
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@p
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_106/DisableCopyOnReadDisableCopyOnRead/read_106_disablecopyonread_adam_m_decodeba_bias"/device:CPU:0*
_output_shapes
 �
Read_106/ReadVariableOpReadVariableOp/read_106_disablecopyonread_adam_m_decodeba_bias^Read_106/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_107/DisableCopyOnReadDisableCopyOnRead/read_107_disablecopyonread_adam_v_decodeba_bias"/device:CPU:0*
_output_shapes
 �
Read_107/ReadVariableOpReadVariableOp/read_107_disablecopyonread_adam_v_decodeba_bias^Read_107/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_108/DisableCopyOnReadDisableCopyOnRead1read_108_disablecopyonread_adam_m_decodebb_kernel"/device:CPU:0*
_output_shapes
 �
Read_108/ReadVariableOpReadVariableOp1read_108_disablecopyonread_adam_m_decodebb_kernel^Read_108/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_109/DisableCopyOnReadDisableCopyOnRead1read_109_disablecopyonread_adam_v_decodebb_kernel"/device:CPU:0*
_output_shapes
 �
Read_109/ReadVariableOpReadVariableOp1read_109_disablecopyonread_adam_v_decodebb_kernel^Read_109/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0y
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_110/DisableCopyOnReadDisableCopyOnRead/read_110_disablecopyonread_adam_m_decodebb_bias"/device:CPU:0*
_output_shapes
 �
Read_110/ReadVariableOpReadVariableOp/read_110_disablecopyonread_adam_m_decodebb_bias^Read_110/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_111/DisableCopyOnReadDisableCopyOnRead/read_111_disablecopyonread_adam_v_decodebb_bias"/device:CPU:0*
_output_shapes
 �
Read_111/ReadVariableOpReadVariableOp/read_111_disablecopyonread_adam_v_decodebb_bias^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0m
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_112/DisableCopyOnReadDisableCopyOnRead3read_112_disablecopyonread_adam_m_transconvb_kernel"/device:CPU:0*
_output_shapes
 �
Read_112/ReadVariableOpReadVariableOp3read_112_disablecopyonread_adam_m_transconvb_kernel^Read_112/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0y
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_113/DisableCopyOnReadDisableCopyOnRead3read_113_disablecopyonread_adam_v_transconvb_kernel"/device:CPU:0*
_output_shapes
 �
Read_113/ReadVariableOpReadVariableOp3read_113_disablecopyonread_adam_v_transconvb_kernel^Read_113/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0y
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_114/DisableCopyOnReadDisableCopyOnRead1read_114_disablecopyonread_adam_m_transconvb_bias"/device:CPU:0*
_output_shapes
 �
Read_114/ReadVariableOpReadVariableOp1read_114_disablecopyonread_adam_m_transconvb_bias^Read_114/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0m
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_115/DisableCopyOnReadDisableCopyOnRead1read_115_disablecopyonread_adam_v_transconvb_bias"/device:CPU:0*
_output_shapes
 �
Read_115/ReadVariableOpReadVariableOp1read_115_disablecopyonread_adam_v_transconvb_bias^Read_115/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0m
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_116/DisableCopyOnReadDisableCopyOnRead1read_116_disablecopyonread_adam_m_decodeaa_kernel"/device:CPU:0*
_output_shapes
 �
Read_116/ReadVariableOpReadVariableOp1read_116_disablecopyonread_adam_m_decodeaa_kernel^Read_116/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0y
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ o
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_117/DisableCopyOnReadDisableCopyOnRead1read_117_disablecopyonread_adam_v_decodeaa_kernel"/device:CPU:0*
_output_shapes
 �
Read_117/ReadVariableOpReadVariableOp1read_117_disablecopyonread_adam_v_decodeaa_kernel^Read_117/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0y
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ o
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_118/DisableCopyOnReadDisableCopyOnRead/read_118_disablecopyonread_adam_m_decodeaa_bias"/device:CPU:0*
_output_shapes
 �
Read_118/ReadVariableOpReadVariableOp/read_118_disablecopyonread_adam_m_decodeaa_bias^Read_118/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0m
Identity_236IdentityRead_118/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_237IdentityIdentity_236:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_119/DisableCopyOnReadDisableCopyOnRead/read_119_disablecopyonread_adam_v_decodeaa_bias"/device:CPU:0*
_output_shapes
 �
Read_119/ReadVariableOpReadVariableOp/read_119_disablecopyonread_adam_v_decodeaa_bias^Read_119/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0m
Identity_238IdentityRead_119/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_239IdentityIdentity_238:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_120/DisableCopyOnReadDisableCopyOnRead1read_120_disablecopyonread_adam_m_decodeab_kernel"/device:CPU:0*
_output_shapes
 �
Read_120/ReadVariableOpReadVariableOp1read_120_disablecopyonread_adam_m_decodeab_kernel^Read_120/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0y
Identity_240IdentityRead_120/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_241IdentityIdentity_240:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_121/DisableCopyOnReadDisableCopyOnRead1read_121_disablecopyonread_adam_v_decodeab_kernel"/device:CPU:0*
_output_shapes
 �
Read_121/ReadVariableOpReadVariableOp1read_121_disablecopyonread_adam_v_decodeab_kernel^Read_121/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0y
Identity_242IdentityRead_121/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_243IdentityIdentity_242:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_122/DisableCopyOnReadDisableCopyOnRead/read_122_disablecopyonread_adam_m_decodeab_bias"/device:CPU:0*
_output_shapes
 �
Read_122/ReadVariableOpReadVariableOp/read_122_disablecopyonread_adam_m_decodeab_bias^Read_122/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0m
Identity_244IdentityRead_122/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_245IdentityIdentity_244:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_123/DisableCopyOnReadDisableCopyOnRead/read_123_disablecopyonread_adam_v_decodeab_bias"/device:CPU:0*
_output_shapes
 �
Read_123/ReadVariableOpReadVariableOp/read_123_disablecopyonread_adam_v_decodeab_bias^Read_123/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0m
Identity_246IdentityRead_123/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_247IdentityIdentity_246:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_124/DisableCopyOnReadDisableCopyOnRead3read_124_disablecopyonread_adam_m_transconva_kernel"/device:CPU:0*
_output_shapes
 �
Read_124/ReadVariableOpReadVariableOp3read_124_disablecopyonread_adam_m_transconva_kernel^Read_124/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0y
Identity_248IdentityRead_124/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_249IdentityIdentity_248:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_125/DisableCopyOnReadDisableCopyOnRead3read_125_disablecopyonread_adam_v_transconva_kernel"/device:CPU:0*
_output_shapes
 �
Read_125/ReadVariableOpReadVariableOp3read_125_disablecopyonread_adam_v_transconva_kernel^Read_125/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0y
Identity_250IdentityRead_125/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_251IdentityIdentity_250:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_126/DisableCopyOnReadDisableCopyOnRead1read_126_disablecopyonread_adam_m_transconva_bias"/device:CPU:0*
_output_shapes
 �
Read_126/ReadVariableOpReadVariableOp1read_126_disablecopyonread_adam_m_transconva_bias^Read_126/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_252IdentityRead_126/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_253IdentityIdentity_252:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_127/DisableCopyOnReadDisableCopyOnRead1read_127_disablecopyonread_adam_v_transconva_bias"/device:CPU:0*
_output_shapes
 �
Read_127/ReadVariableOpReadVariableOp1read_127_disablecopyonread_adam_v_transconva_bias^Read_127/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_254IdentityRead_127/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_255IdentityIdentity_254:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_128/DisableCopyOnReadDisableCopyOnRead1read_128_disablecopyonread_adam_m_convouta_kernel"/device:CPU:0*
_output_shapes
 �
Read_128/ReadVariableOpReadVariableOp1read_128_disablecopyonread_adam_m_convouta_kernel^Read_128/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0y
Identity_256IdentityRead_128/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_257IdentityIdentity_256:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_129/DisableCopyOnReadDisableCopyOnRead1read_129_disablecopyonread_adam_v_convouta_kernel"/device:CPU:0*
_output_shapes
 �
Read_129/ReadVariableOpReadVariableOp1read_129_disablecopyonread_adam_v_convouta_kernel^Read_129/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0y
Identity_258IdentityRead_129/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_259IdentityIdentity_258:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_130/DisableCopyOnReadDisableCopyOnRead/read_130_disablecopyonread_adam_m_convouta_bias"/device:CPU:0*
_output_shapes
 �
Read_130/ReadVariableOpReadVariableOp/read_130_disablecopyonread_adam_m_convouta_bias^Read_130/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_260IdentityRead_130/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_261IdentityIdentity_260:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_131/DisableCopyOnReadDisableCopyOnRead/read_131_disablecopyonread_adam_v_convouta_bias"/device:CPU:0*
_output_shapes
 �
Read_131/ReadVariableOpReadVariableOp/read_131_disablecopyonread_adam_v_convouta_bias^Read_131/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_262IdentityRead_131/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_263IdentityIdentity_262:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_132/DisableCopyOnReadDisableCopyOnRead1read_132_disablecopyonread_adam_m_convoutb_kernel"/device:CPU:0*
_output_shapes
 �
Read_132/ReadVariableOpReadVariableOp1read_132_disablecopyonread_adam_m_convoutb_kernel^Read_132/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_264IdentityRead_132/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_265IdentityIdentity_264:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_133/DisableCopyOnReadDisableCopyOnRead1read_133_disablecopyonread_adam_v_convoutb_kernel"/device:CPU:0*
_output_shapes
 �
Read_133/ReadVariableOpReadVariableOp1read_133_disablecopyonread_adam_v_convoutb_kernel^Read_133/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_266IdentityRead_133/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_267IdentityIdentity_266:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_134/DisableCopyOnReadDisableCopyOnRead/read_134_disablecopyonread_adam_m_convoutb_bias"/device:CPU:0*
_output_shapes
 �
Read_134/ReadVariableOpReadVariableOp/read_134_disablecopyonread_adam_m_convoutb_bias^Read_134/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_268IdentityRead_134/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_269IdentityIdentity_268:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_135/DisableCopyOnReadDisableCopyOnRead/read_135_disablecopyonread_adam_v_convoutb_bias"/device:CPU:0*
_output_shapes
 �
Read_135/ReadVariableOpReadVariableOp/read_135_disablecopyonread_adam_v_convoutb_bias^Read_135/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_270IdentityRead_135/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_271IdentityIdentity_270:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_136/DisableCopyOnReadDisableCopyOnRead7read_136_disablecopyonread_adam_m_predictionmask_kernel"/device:CPU:0*
_output_shapes
 �
Read_136/ReadVariableOpReadVariableOp7read_136_disablecopyonread_adam_m_predictionmask_kernel^Read_136/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_272IdentityRead_136/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_273IdentityIdentity_272:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_137/DisableCopyOnReadDisableCopyOnRead7read_137_disablecopyonread_adam_v_predictionmask_kernel"/device:CPU:0*
_output_shapes
 �
Read_137/ReadVariableOpReadVariableOp7read_137_disablecopyonread_adam_v_predictionmask_kernel^Read_137/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0y
Identity_274IdentityRead_137/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_275IdentityIdentity_274:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_138/DisableCopyOnReadDisableCopyOnRead5read_138_disablecopyonread_adam_m_predictionmask_bias"/device:CPU:0*
_output_shapes
 �
Read_138/ReadVariableOpReadVariableOp5read_138_disablecopyonread_adam_m_predictionmask_bias^Read_138/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_276IdentityRead_138/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_277IdentityIdentity_276:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_139/DisableCopyOnReadDisableCopyOnRead5read_139_disablecopyonread_adam_v_predictionmask_bias"/device:CPU:0*
_output_shapes
 �
Read_139/ReadVariableOpReadVariableOp5read_139_disablecopyonread_adam_v_predictionmask_bias^Read_139/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_278IdentityRead_139/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_279IdentityIdentity_278:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_140/DisableCopyOnReadDisableCopyOnRead"read_140_disablecopyonread_total_6"/device:CPU:0*
_output_shapes
 �
Read_140/ReadVariableOpReadVariableOp"read_140_disablecopyonread_total_6^Read_140/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_280IdentityRead_140/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_281IdentityIdentity_280:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_141/DisableCopyOnReadDisableCopyOnRead"read_141_disablecopyonread_count_6"/device:CPU:0*
_output_shapes
 �
Read_141/ReadVariableOpReadVariableOp"read_141_disablecopyonread_count_6^Read_141/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_282IdentityRead_141/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_283IdentityIdentity_282:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_142/DisableCopyOnReadDisableCopyOnRead"read_142_disablecopyonread_total_5"/device:CPU:0*
_output_shapes
 �
Read_142/ReadVariableOpReadVariableOp"read_142_disablecopyonread_total_5^Read_142/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_284IdentityRead_142/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_285IdentityIdentity_284:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_143/DisableCopyOnReadDisableCopyOnRead"read_143_disablecopyonread_count_5"/device:CPU:0*
_output_shapes
 �
Read_143/ReadVariableOpReadVariableOp"read_143_disablecopyonread_count_5^Read_143/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_286IdentityRead_143/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_287IdentityIdentity_286:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_144/DisableCopyOnReadDisableCopyOnRead"read_144_disablecopyonread_total_4"/device:CPU:0*
_output_shapes
 �
Read_144/ReadVariableOpReadVariableOp"read_144_disablecopyonread_total_4^Read_144/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_288IdentityRead_144/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_289IdentityIdentity_288:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_145/DisableCopyOnReadDisableCopyOnRead"read_145_disablecopyonread_count_4"/device:CPU:0*
_output_shapes
 �
Read_145/ReadVariableOpReadVariableOp"read_145_disablecopyonread_count_4^Read_145/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_290IdentityRead_145/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_291IdentityIdentity_290:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_146/DisableCopyOnReadDisableCopyOnRead"read_146_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 �
Read_146/ReadVariableOpReadVariableOp"read_146_disablecopyonread_total_3^Read_146/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_292IdentityRead_146/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_293IdentityIdentity_292:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_147/DisableCopyOnReadDisableCopyOnRead"read_147_disablecopyonread_count_3"/device:CPU:0*
_output_shapes
 �
Read_147/ReadVariableOpReadVariableOp"read_147_disablecopyonread_count_3^Read_147/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_294IdentityRead_147/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_295IdentityIdentity_294:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_148/DisableCopyOnReadDisableCopyOnRead"read_148_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_148/ReadVariableOpReadVariableOp"read_148_disablecopyonread_total_2^Read_148/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_296IdentityRead_148/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_297IdentityIdentity_296:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_149/DisableCopyOnReadDisableCopyOnRead"read_149_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_149/ReadVariableOpReadVariableOp"read_149_disablecopyonread_count_2^Read_149/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_298IdentityRead_149/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_299IdentityIdentity_298:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_150/DisableCopyOnReadDisableCopyOnRead"read_150_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_150/ReadVariableOpReadVariableOp"read_150_disablecopyonread_total_1^Read_150/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_300IdentityRead_150/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_301IdentityIdentity_300:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_151/DisableCopyOnReadDisableCopyOnRead"read_151_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_151/ReadVariableOpReadVariableOp"read_151_disablecopyonread_count_1^Read_151/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_302IdentityRead_151/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_303IdentityIdentity_302:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_152/DisableCopyOnReadDisableCopyOnRead read_152_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_152/ReadVariableOpReadVariableOp read_152_disablecopyonread_total^Read_152/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_304IdentityRead_152/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_305IdentityIdentity_304:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_153/DisableCopyOnReadDisableCopyOnRead read_153_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_153/ReadVariableOpReadVariableOp read_153_disablecopyonread_count^Read_153/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_306IdentityRead_153/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_307IdentityIdentity_306:output:0"/device:CPU:0*
T0*
_output_shapes
: �A
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�@
value�@B�@�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0Identity_237:output:0Identity_239:output:0Identity_241:output:0Identity_243:output:0Identity_245:output:0Identity_247:output:0Identity_249:output:0Identity_251:output:0Identity_253:output:0Identity_255:output:0Identity_257:output:0Identity_259:output:0Identity_261:output:0Identity_263:output:0Identity_265:output:0Identity_267:output:0Identity_269:output:0Identity_271:output:0Identity_273:output:0Identity_275:output:0Identity_277:output:0Identity_279:output:0Identity_281:output:0Identity_283:output:0Identity_285:output:0Identity_287:output:0Identity_289:output:0Identity_291:output:0Identity_293:output:0Identity_295:output:0Identity_297:output:0Identity_299:output:0Identity_301:output:0Identity_303:output:0Identity_305:output:0Identity_307:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *�
dtypes�
�2�	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_308Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_309IdentityIdentity_308:output:0^NoOp*
T0*
_output_shapes
: �@
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_118/DisableCopyOnRead^Read_118/ReadVariableOp^Read_119/DisableCopyOnRead^Read_119/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_120/DisableCopyOnRead^Read_120/ReadVariableOp^Read_121/DisableCopyOnRead^Read_121/ReadVariableOp^Read_122/DisableCopyOnRead^Read_122/ReadVariableOp^Read_123/DisableCopyOnRead^Read_123/ReadVariableOp^Read_124/DisableCopyOnRead^Read_124/ReadVariableOp^Read_125/DisableCopyOnRead^Read_125/ReadVariableOp^Read_126/DisableCopyOnRead^Read_126/ReadVariableOp^Read_127/DisableCopyOnRead^Read_127/ReadVariableOp^Read_128/DisableCopyOnRead^Read_128/ReadVariableOp^Read_129/DisableCopyOnRead^Read_129/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_130/DisableCopyOnRead^Read_130/ReadVariableOp^Read_131/DisableCopyOnRead^Read_131/ReadVariableOp^Read_132/DisableCopyOnRead^Read_132/ReadVariableOp^Read_133/DisableCopyOnRead^Read_133/ReadVariableOp^Read_134/DisableCopyOnRead^Read_134/ReadVariableOp^Read_135/DisableCopyOnRead^Read_135/ReadVariableOp^Read_136/DisableCopyOnRead^Read_136/ReadVariableOp^Read_137/DisableCopyOnRead^Read_137/ReadVariableOp^Read_138/DisableCopyOnRead^Read_138/ReadVariableOp^Read_139/DisableCopyOnRead^Read_139/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_140/DisableCopyOnRead^Read_140/ReadVariableOp^Read_141/DisableCopyOnRead^Read_141/ReadVariableOp^Read_142/DisableCopyOnRead^Read_142/ReadVariableOp^Read_143/DisableCopyOnRead^Read_143/ReadVariableOp^Read_144/DisableCopyOnRead^Read_144/ReadVariableOp^Read_145/DisableCopyOnRead^Read_145/ReadVariableOp^Read_146/DisableCopyOnRead^Read_146/ReadVariableOp^Read_147/DisableCopyOnRead^Read_147/ReadVariableOp^Read_148/DisableCopyOnRead^Read_148/ReadVariableOp^Read_149/DisableCopyOnRead^Read_149/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_150/DisableCopyOnRead^Read_150/ReadVariableOp^Read_151/DisableCopyOnRead^Read_151/ReadVariableOp^Read_152/DisableCopyOnRead^Read_152/ReadVariableOp^Read_153/DisableCopyOnRead^Read_153/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_309Identity_309:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp28
Read_118/DisableCopyOnReadRead_118/DisableCopyOnRead22
Read_118/ReadVariableOpRead_118/ReadVariableOp28
Read_119/DisableCopyOnReadRead_119/DisableCopyOnRead22
Read_119/ReadVariableOpRead_119/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp28
Read_120/DisableCopyOnReadRead_120/DisableCopyOnRead22
Read_120/ReadVariableOpRead_120/ReadVariableOp28
Read_121/DisableCopyOnReadRead_121/DisableCopyOnRead22
Read_121/ReadVariableOpRead_121/ReadVariableOp28
Read_122/DisableCopyOnReadRead_122/DisableCopyOnRead22
Read_122/ReadVariableOpRead_122/ReadVariableOp28
Read_123/DisableCopyOnReadRead_123/DisableCopyOnRead22
Read_123/ReadVariableOpRead_123/ReadVariableOp28
Read_124/DisableCopyOnReadRead_124/DisableCopyOnRead22
Read_124/ReadVariableOpRead_124/ReadVariableOp28
Read_125/DisableCopyOnReadRead_125/DisableCopyOnRead22
Read_125/ReadVariableOpRead_125/ReadVariableOp28
Read_126/DisableCopyOnReadRead_126/DisableCopyOnRead22
Read_126/ReadVariableOpRead_126/ReadVariableOp28
Read_127/DisableCopyOnReadRead_127/DisableCopyOnRead22
Read_127/ReadVariableOpRead_127/ReadVariableOp28
Read_128/DisableCopyOnReadRead_128/DisableCopyOnRead22
Read_128/ReadVariableOpRead_128/ReadVariableOp28
Read_129/DisableCopyOnReadRead_129/DisableCopyOnRead22
Read_129/ReadVariableOpRead_129/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp28
Read_130/DisableCopyOnReadRead_130/DisableCopyOnRead22
Read_130/ReadVariableOpRead_130/ReadVariableOp28
Read_131/DisableCopyOnReadRead_131/DisableCopyOnRead22
Read_131/ReadVariableOpRead_131/ReadVariableOp28
Read_132/DisableCopyOnReadRead_132/DisableCopyOnRead22
Read_132/ReadVariableOpRead_132/ReadVariableOp28
Read_133/DisableCopyOnReadRead_133/DisableCopyOnRead22
Read_133/ReadVariableOpRead_133/ReadVariableOp28
Read_134/DisableCopyOnReadRead_134/DisableCopyOnRead22
Read_134/ReadVariableOpRead_134/ReadVariableOp28
Read_135/DisableCopyOnReadRead_135/DisableCopyOnRead22
Read_135/ReadVariableOpRead_135/ReadVariableOp28
Read_136/DisableCopyOnReadRead_136/DisableCopyOnRead22
Read_136/ReadVariableOpRead_136/ReadVariableOp28
Read_137/DisableCopyOnReadRead_137/DisableCopyOnRead22
Read_137/ReadVariableOpRead_137/ReadVariableOp28
Read_138/DisableCopyOnReadRead_138/DisableCopyOnRead22
Read_138/ReadVariableOpRead_138/ReadVariableOp28
Read_139/DisableCopyOnReadRead_139/DisableCopyOnRead22
Read_139/ReadVariableOpRead_139/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp28
Read_140/DisableCopyOnReadRead_140/DisableCopyOnRead22
Read_140/ReadVariableOpRead_140/ReadVariableOp28
Read_141/DisableCopyOnReadRead_141/DisableCopyOnRead22
Read_141/ReadVariableOpRead_141/ReadVariableOp28
Read_142/DisableCopyOnReadRead_142/DisableCopyOnRead22
Read_142/ReadVariableOpRead_142/ReadVariableOp28
Read_143/DisableCopyOnReadRead_143/DisableCopyOnRead22
Read_143/ReadVariableOpRead_143/ReadVariableOp28
Read_144/DisableCopyOnReadRead_144/DisableCopyOnRead22
Read_144/ReadVariableOpRead_144/ReadVariableOp28
Read_145/DisableCopyOnReadRead_145/DisableCopyOnRead22
Read_145/ReadVariableOpRead_145/ReadVariableOp28
Read_146/DisableCopyOnReadRead_146/DisableCopyOnRead22
Read_146/ReadVariableOpRead_146/ReadVariableOp28
Read_147/DisableCopyOnReadRead_147/DisableCopyOnRead22
Read_147/ReadVariableOpRead_147/ReadVariableOp28
Read_148/DisableCopyOnReadRead_148/DisableCopyOnRead22
Read_148/ReadVariableOpRead_148/ReadVariableOp28
Read_149/DisableCopyOnReadRead_149/DisableCopyOnRead22
Read_149/ReadVariableOpRead_149/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp28
Read_150/DisableCopyOnReadRead_150/DisableCopyOnRead22
Read_150/ReadVariableOpRead_150/ReadVariableOp28
Read_151/DisableCopyOnReadRead_151/DisableCopyOnRead22
Read_151/ReadVariableOpRead_151/ReadVariableOp28
Read_152/DisableCopyOnReadRead_152/DisableCopyOnRead22
Read_152/ReadVariableOpRead_152/ReadVariableOp28
Read_153/DisableCopyOnReadRead_153/DisableCopyOnRead22
Read_153/ReadVariableOpRead_153/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameencodeAa/kernel:-)
'
_user_specified_nameencodeAa/bias:/+
)
_user_specified_nameencodeAb/kernel:-)
'
_user_specified_nameencodeAb/bias:/+
)
_user_specified_nameencodeBa/kernel:-)
'
_user_specified_nameencodeBa/bias:/+
)
_user_specified_nameencodeBb/kernel:-)
'
_user_specified_nameencodeBb/bias:/	+
)
_user_specified_nameencodeCa/kernel:-
)
'
_user_specified_nameencodeCa/bias:/+
)
_user_specified_nameencodeCb/kernel:-)
'
_user_specified_nameencodeCb/bias:/+
)
_user_specified_nameencodeDa/kernel:-)
'
_user_specified_nameencodeDa/bias:/+
)
_user_specified_nameencodeDb/kernel:-)
'
_user_specified_nameencodeDb/bias:/+
)
_user_specified_nameencodeEa/kernel:-)
'
_user_specified_nameencodeEa/bias:/+
)
_user_specified_nameencodeEb/kernel:-)
'
_user_specified_nameencodeEb/bias:1-
+
_user_specified_nametransconvE/kernel:/+
)
_user_specified_nametransconvE/bias:/+
)
_user_specified_namedecodeCa/kernel:-)
'
_user_specified_namedecodeCa/bias:/+
)
_user_specified_namedecodeCb/kernel:-)
'
_user_specified_namedecodeCb/bias:1-
+
_user_specified_nametransconvC/kernel:/+
)
_user_specified_nametransconvC/bias:/+
)
_user_specified_namedecodeBa/kernel:-)
'
_user_specified_namedecodeBa/bias:/+
)
_user_specified_namedecodeBb/kernel:- )
'
_user_specified_namedecodeBb/bias:1!-
+
_user_specified_nametransconvB/kernel:/"+
)
_user_specified_nametransconvB/bias:/#+
)
_user_specified_namedecodeAa/kernel:-$)
'
_user_specified_namedecodeAa/bias:/%+
)
_user_specified_namedecodeAb/kernel:-&)
'
_user_specified_namedecodeAb/bias:1'-
+
_user_specified_nametransconvA/kernel:/(+
)
_user_specified_nametransconvA/bias:/)+
)
_user_specified_nameconvOuta/kernel:-*)
'
_user_specified_nameconvOuta/bias:/++
)
_user_specified_nameconvOutb/kernel:-,)
'
_user_specified_nameconvOutb/bias:5-1
/
_user_specified_namePredictionMask/kernel:3./
-
_user_specified_namePredictionMask/bias:)/%
#
_user_specified_name	iteration:-0)
'
_user_specified_namelearning_rate:612
0
_user_specified_nameAdam/m/encodeAa/kernel:622
0
_user_specified_nameAdam/v/encodeAa/kernel:430
.
_user_specified_nameAdam/m/encodeAa/bias:440
.
_user_specified_nameAdam/v/encodeAa/bias:652
0
_user_specified_nameAdam/m/encodeAb/kernel:662
0
_user_specified_nameAdam/v/encodeAb/kernel:470
.
_user_specified_nameAdam/m/encodeAb/bias:480
.
_user_specified_nameAdam/v/encodeAb/bias:692
0
_user_specified_nameAdam/m/encodeBa/kernel:6:2
0
_user_specified_nameAdam/v/encodeBa/kernel:4;0
.
_user_specified_nameAdam/m/encodeBa/bias:4<0
.
_user_specified_nameAdam/v/encodeBa/bias:6=2
0
_user_specified_nameAdam/m/encodeBb/kernel:6>2
0
_user_specified_nameAdam/v/encodeBb/kernel:4?0
.
_user_specified_nameAdam/m/encodeBb/bias:4@0
.
_user_specified_nameAdam/v/encodeBb/bias:6A2
0
_user_specified_nameAdam/m/encodeCa/kernel:6B2
0
_user_specified_nameAdam/v/encodeCa/kernel:4C0
.
_user_specified_nameAdam/m/encodeCa/bias:4D0
.
_user_specified_nameAdam/v/encodeCa/bias:6E2
0
_user_specified_nameAdam/m/encodeCb/kernel:6F2
0
_user_specified_nameAdam/v/encodeCb/kernel:4G0
.
_user_specified_nameAdam/m/encodeCb/bias:4H0
.
_user_specified_nameAdam/v/encodeCb/bias:6I2
0
_user_specified_nameAdam/m/encodeDa/kernel:6J2
0
_user_specified_nameAdam/v/encodeDa/kernel:4K0
.
_user_specified_nameAdam/m/encodeDa/bias:4L0
.
_user_specified_nameAdam/v/encodeDa/bias:6M2
0
_user_specified_nameAdam/m/encodeDb/kernel:6N2
0
_user_specified_nameAdam/v/encodeDb/kernel:4O0
.
_user_specified_nameAdam/m/encodeDb/bias:4P0
.
_user_specified_nameAdam/v/encodeDb/bias:6Q2
0
_user_specified_nameAdam/m/encodeEa/kernel:6R2
0
_user_specified_nameAdam/v/encodeEa/kernel:4S0
.
_user_specified_nameAdam/m/encodeEa/bias:4T0
.
_user_specified_nameAdam/v/encodeEa/bias:6U2
0
_user_specified_nameAdam/m/encodeEb/kernel:6V2
0
_user_specified_nameAdam/v/encodeEb/kernel:4W0
.
_user_specified_nameAdam/m/encodeEb/bias:4X0
.
_user_specified_nameAdam/v/encodeEb/bias:8Y4
2
_user_specified_nameAdam/m/transconvE/kernel:8Z4
2
_user_specified_nameAdam/v/transconvE/kernel:6[2
0
_user_specified_nameAdam/m/transconvE/bias:6\2
0
_user_specified_nameAdam/v/transconvE/bias:6]2
0
_user_specified_nameAdam/m/decodeCa/kernel:6^2
0
_user_specified_nameAdam/v/decodeCa/kernel:4_0
.
_user_specified_nameAdam/m/decodeCa/bias:4`0
.
_user_specified_nameAdam/v/decodeCa/bias:6a2
0
_user_specified_nameAdam/m/decodeCb/kernel:6b2
0
_user_specified_nameAdam/v/decodeCb/kernel:4c0
.
_user_specified_nameAdam/m/decodeCb/bias:4d0
.
_user_specified_nameAdam/v/decodeCb/bias:8e4
2
_user_specified_nameAdam/m/transconvC/kernel:8f4
2
_user_specified_nameAdam/v/transconvC/kernel:6g2
0
_user_specified_nameAdam/m/transconvC/bias:6h2
0
_user_specified_nameAdam/v/transconvC/bias:6i2
0
_user_specified_nameAdam/m/decodeBa/kernel:6j2
0
_user_specified_nameAdam/v/decodeBa/kernel:4k0
.
_user_specified_nameAdam/m/decodeBa/bias:4l0
.
_user_specified_nameAdam/v/decodeBa/bias:6m2
0
_user_specified_nameAdam/m/decodeBb/kernel:6n2
0
_user_specified_nameAdam/v/decodeBb/kernel:4o0
.
_user_specified_nameAdam/m/decodeBb/bias:4p0
.
_user_specified_nameAdam/v/decodeBb/bias:8q4
2
_user_specified_nameAdam/m/transconvB/kernel:8r4
2
_user_specified_nameAdam/v/transconvB/kernel:6s2
0
_user_specified_nameAdam/m/transconvB/bias:6t2
0
_user_specified_nameAdam/v/transconvB/bias:6u2
0
_user_specified_nameAdam/m/decodeAa/kernel:6v2
0
_user_specified_nameAdam/v/decodeAa/kernel:4w0
.
_user_specified_nameAdam/m/decodeAa/bias:4x0
.
_user_specified_nameAdam/v/decodeAa/bias:6y2
0
_user_specified_nameAdam/m/decodeAb/kernel:6z2
0
_user_specified_nameAdam/v/decodeAb/kernel:4{0
.
_user_specified_nameAdam/m/decodeAb/bias:4|0
.
_user_specified_nameAdam/v/decodeAb/bias:8}4
2
_user_specified_nameAdam/m/transconvA/kernel:8~4
2
_user_specified_nameAdam/v/transconvA/kernel:62
0
_user_specified_nameAdam/m/transconvA/bias:7�2
0
_user_specified_nameAdam/v/transconvA/bias:7�2
0
_user_specified_nameAdam/m/convOuta/kernel:7�2
0
_user_specified_nameAdam/v/convOuta/kernel:5�0
.
_user_specified_nameAdam/m/convOuta/bias:5�0
.
_user_specified_nameAdam/v/convOuta/bias:7�2
0
_user_specified_nameAdam/m/convOutb/kernel:7�2
0
_user_specified_nameAdam/v/convOutb/kernel:5�0
.
_user_specified_nameAdam/m/convOutb/bias:5�0
.
_user_specified_nameAdam/v/convOutb/bias:=�8
6
_user_specified_nameAdam/m/PredictionMask/kernel:=�8
6
_user_specified_nameAdam/v/PredictionMask/kernel:;�6
4
_user_specified_nameAdam/m/PredictionMask/bias:;�6
4
_user_specified_nameAdam/v/PredictionMask/bias:(�#
!
_user_specified_name	total_6:(�#
!
_user_specified_name	count_6:(�#
!
_user_specified_name	total_5:(�#
!
_user_specified_name	count_5:(�#
!
_user_specified_name	total_4:(�#
!
_user_specified_name	count_4:(�#
!
_user_specified_name	total_3:(�#
!
_user_specified_name	count_3:(�#
!
_user_specified_name	total_2:(�#
!
_user_specified_name	count_2:(�#
!
_user_specified_name	total_1:(�#
!
_user_specified_name	count_1:&�!

_user_specified_nametotal:&�!

_user_specified_namecount:>�9

_output_shapes
: 

_user_specified_nameConst
�&
�
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191591
mrimages!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@#
	unknown_9:@@

unknown_10:@%

unknown_11:@�

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�%

unknown_25:@�

unknown_26:@%

unknown_27:�@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31: @

unknown_32: $

unknown_33:@ 

unknown_34: $

unknown_35:  

unknown_36: $

unknown_37: 

unknown_38:$

unknown_39: 

unknown_40:$

unknown_41:

unknown_42:$

unknown_43:

unknown_44:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmrimagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*P
_read_only_resource_inputs2
0.	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191365y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes{
y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages:&"
 
_user_specified_name191497:&"
 
_user_specified_name191499:&"
 
_user_specified_name191501:&"
 
_user_specified_name191503:&"
 
_user_specified_name191505:&"
 
_user_specified_name191507:&"
 
_user_specified_name191509:&"
 
_user_specified_name191511:&	"
 
_user_specified_name191513:&
"
 
_user_specified_name191515:&"
 
_user_specified_name191517:&"
 
_user_specified_name191519:&"
 
_user_specified_name191521:&"
 
_user_specified_name191523:&"
 
_user_specified_name191525:&"
 
_user_specified_name191527:&"
 
_user_specified_name191529:&"
 
_user_specified_name191531:&"
 
_user_specified_name191533:&"
 
_user_specified_name191535:&"
 
_user_specified_name191537:&"
 
_user_specified_name191539:&"
 
_user_specified_name191541:&"
 
_user_specified_name191543:&"
 
_user_specified_name191545:&"
 
_user_specified_name191547:&"
 
_user_specified_name191549:&"
 
_user_specified_name191551:&"
 
_user_specified_name191553:&"
 
_user_specified_name191555:&"
 
_user_specified_name191557:& "
 
_user_specified_name191559:&!"
 
_user_specified_name191561:&""
 
_user_specified_name191563:&#"
 
_user_specified_name191565:&$"
 
_user_specified_name191567:&%"
 
_user_specified_name191569:&&"
 
_user_specified_name191571:&'"
 
_user_specified_name191573:&("
 
_user_specified_name191575:&)"
 
_user_specified_name191577:&*"
 
_user_specified_name191579:&+"
 
_user_specified_name191581:&,"
 
_user_specified_name191583:&-"
 
_user_specified_name191585:&."
 
_user_specified_name191587
�
�
D__inference_encodeEa_layer_call_and_return_conditional_losses_191146

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
]
A__inference_poolD_layer_call_and_return_conditional_losses_192259

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_convOutb_layer_call_fn_192668

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_convOutb_layer_call_and_return_conditional_losses_191342y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192662:&"
 
_user_specified_name192664
�
�
)__inference_encodeAa_layer_call_fn_191992

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeAa_layer_call_and_return_conditional_losses_191012y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:&"
 
_user_specified_name191986:&"
 
_user_specified_name191988
ť
�
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191365
mrimages)
encodeaa_191013:
encodeaa_191015:)
encodeab_191029:
encodeab_191031:)
encodeba_191046: 
encodeba_191048: )
encodebb_191062:  
encodebb_191064: )
encodeca_191079: @
encodeca_191081:@)
encodecb_191096:@@
encodecb_191098:@*
encodeda_191113:@�
encodeda_191115:	�+
encodedb_191130:��
encodedb_191132:	�+
encodeea_191147:��
encodeea_191149:	�+
encodeeb_191163:��
encodeeb_191165:	�-
transconve_191168:�� 
transconve_191170:	�+
decodeca_191192:��
decodeca_191194:	�+
decodecb_191208:��
decodecb_191210:	�,
transconvc_191213:@�
transconvc_191215:@*
decodeba_191237:�@
decodeba_191239:@)
decodebb_191253:@@
decodebb_191255:@+
transconvb_191258: @
transconvb_191260: )
decodeaa_191282:@ 
decodeaa_191284: )
decodeab_191298:  
decodeab_191300: +
transconva_191303: 
transconva_191305:)
convouta_191327: 
convouta_191329:)
convoutb_191343:
convoutb_191345:/
predictionmask_191359:#
predictionmask_191361:
identity��&PredictionMask/StatefulPartitionedCall� convOuta/StatefulPartitionedCall� convOutb/StatefulPartitionedCall� decodeAa/StatefulPartitionedCall� decodeAb/StatefulPartitionedCall� decodeBa/StatefulPartitionedCall� decodeBb/StatefulPartitionedCall� decodeCa/StatefulPartitionedCall� decodeCb/StatefulPartitionedCall� encodeAa/StatefulPartitionedCall� encodeAb/StatefulPartitionedCall� encodeBa/StatefulPartitionedCall� encodeBb/StatefulPartitionedCall� encodeCa/StatefulPartitionedCall� encodeCb/StatefulPartitionedCall� encodeDa/StatefulPartitionedCall� encodeDb/StatefulPartitionedCall� encodeEa/StatefulPartitionedCall� encodeEb/StatefulPartitionedCall�)spatial_dropout2d/StatefulPartitionedCall�+spatial_dropout2d_1/StatefulPartitionedCall�"transconvA/StatefulPartitionedCall�"transconvB/StatefulPartitionedCall�"transconvC/StatefulPartitionedCall�"transconvE/StatefulPartitionedCall�
 encodeAa/StatefulPartitionedCallStatefulPartitionedCallmrimagesencodeaa_191013encodeaa_191015*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeAa_layer_call_and_return_conditional_losses_191012�
 encodeAb/StatefulPartitionedCallStatefulPartitionedCall)encodeAa/StatefulPartitionedCall:output:0encodeab_191029encodeab_191031*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeAb_layer_call_and_return_conditional_losses_191028�
poolA/PartitionedCallPartitionedCall)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolA_layer_call_and_return_conditional_losses_190720�
 encodeBa/StatefulPartitionedCallStatefulPartitionedCallpoolA/PartitionedCall:output:0encodeba_191046encodeba_191048*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeBa_layer_call_and_return_conditional_losses_191045�
 encodeBb/StatefulPartitionedCallStatefulPartitionedCall)encodeBa/StatefulPartitionedCall:output:0encodebb_191062encodebb_191064*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeBb_layer_call_and_return_conditional_losses_191061�
poolB/PartitionedCallPartitionedCall)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolB_layer_call_and_return_conditional_losses_190730�
 encodeCa/StatefulPartitionedCallStatefulPartitionedCallpoolB/PartitionedCall:output:0encodeca_191079encodeca_191081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeCa_layer_call_and_return_conditional_losses_191078�
)spatial_dropout2d/StatefulPartitionedCallStatefulPartitionedCall)encodeCa/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_190758�
 encodeCb/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout2d/StatefulPartitionedCall:output:0encodecb_191096encodecb_191098*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeCb_layer_call_and_return_conditional_losses_191095�
poolC/PartitionedCallPartitionedCall)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolC_layer_call_and_return_conditional_losses_190778�
 encodeDa/StatefulPartitionedCallStatefulPartitionedCallpoolC/PartitionedCall:output:0encodeda_191113encodeda_191115*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeDa_layer_call_and_return_conditional_losses_191112�
+spatial_dropout2d_1/StatefulPartitionedCallStatefulPartitionedCall)encodeDa/StatefulPartitionedCall:output:0*^spatial_dropout2d/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_190806�
 encodeDb/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout2d_1/StatefulPartitionedCall:output:0encodedb_191130encodedb_191132*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeDb_layer_call_and_return_conditional_losses_191129�
poolD/PartitionedCallPartitionedCall)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_poolD_layer_call_and_return_conditional_losses_190826�
 encodeEa/StatefulPartitionedCallStatefulPartitionedCallpoolD/PartitionedCall:output:0encodeea_191147encodeea_191149*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeEa_layer_call_and_return_conditional_losses_191146�
 encodeEb/StatefulPartitionedCallStatefulPartitionedCall)encodeEa/StatefulPartitionedCall:output:0encodeeb_191163encodeeb_191165*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeEb_layer_call_and_return_conditional_losses_191162�
"transconvE/StatefulPartitionedCallStatefulPartitionedCall)encodeEb/StatefulPartitionedCall:output:0transconve_191168transconve_191170*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvE_layer_call_and_return_conditional_losses_190864�
concatD/PartitionedCallPartitionedCall+transconvE/StatefulPartitionedCall:output:0)encodeDb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatD_layer_call_and_return_conditional_losses_191179�
 decodeCa/StatefulPartitionedCallStatefulPartitionedCall concatD/PartitionedCall:output:0decodeca_191192decodeca_191194*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeCa_layer_call_and_return_conditional_losses_191191�
 decodeCb/StatefulPartitionedCallStatefulPartitionedCall)decodeCa/StatefulPartitionedCall:output:0decodecb_191208decodecb_191210*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeCb_layer_call_and_return_conditional_losses_191207�
"transconvC/StatefulPartitionedCallStatefulPartitionedCall)decodeCb/StatefulPartitionedCall:output:0transconvc_191213transconvc_191215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvC_layer_call_and_return_conditional_losses_190906�
concatC/PartitionedCallPartitionedCall+transconvC/StatefulPartitionedCall:output:0)encodeCb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatC_layer_call_and_return_conditional_losses_191224�
 decodeBa/StatefulPartitionedCallStatefulPartitionedCall concatC/PartitionedCall:output:0decodeba_191237decodeba_191239*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeBa_layer_call_and_return_conditional_losses_191236�
 decodeBb/StatefulPartitionedCallStatefulPartitionedCall)decodeBa/StatefulPartitionedCall:output:0decodebb_191253decodebb_191255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeBb_layer_call_and_return_conditional_losses_191252�
"transconvB/StatefulPartitionedCallStatefulPartitionedCall)decodeBb/StatefulPartitionedCall:output:0transconvb_191258transconvb_191260*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvB_layer_call_and_return_conditional_losses_190948�
concatB/PartitionedCallPartitionedCall+transconvB/StatefulPartitionedCall:output:0)encodeBb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatB_layer_call_and_return_conditional_losses_191269�
 decodeAa/StatefulPartitionedCallStatefulPartitionedCall concatB/PartitionedCall:output:0decodeaa_191282decodeaa_191284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeAa_layer_call_and_return_conditional_losses_191281�
 decodeAb/StatefulPartitionedCallStatefulPartitionedCall)decodeAa/StatefulPartitionedCall:output:0decodeab_191298decodeab_191300*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_decodeAb_layer_call_and_return_conditional_losses_191297�
"transconvA/StatefulPartitionedCallStatefulPartitionedCall)decodeAb/StatefulPartitionedCall:output:0transconva_191303transconva_191305*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvA_layer_call_and_return_conditional_losses_190990�
concatA/PartitionedCallPartitionedCall+transconvA/StatefulPartitionedCall:output:0)encodeAb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_concatA_layer_call_and_return_conditional_losses_191314�
 convOuta/StatefulPartitionedCallStatefulPartitionedCall concatA/PartitionedCall:output:0convouta_191327convouta_191329*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_convOuta_layer_call_and_return_conditional_losses_191326�
 convOutb/StatefulPartitionedCallStatefulPartitionedCall)convOuta/StatefulPartitionedCall:output:0convoutb_191343convoutb_191345*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_convOutb_layer_call_and_return_conditional_losses_191342�
&PredictionMask/StatefulPartitionedCallStatefulPartitionedCall)convOutb/StatefulPartitionedCall:output:0predictionmask_191359predictionmask_191361*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_PredictionMask_layer_call_and_return_conditional_losses_191358�
IdentityIdentity/PredictionMask/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp'^PredictionMask/StatefulPartitionedCall!^convOuta/StatefulPartitionedCall!^convOutb/StatefulPartitionedCall!^decodeAa/StatefulPartitionedCall!^decodeAb/StatefulPartitionedCall!^decodeBa/StatefulPartitionedCall!^decodeBb/StatefulPartitionedCall!^decodeCa/StatefulPartitionedCall!^decodeCb/StatefulPartitionedCall!^encodeAa/StatefulPartitionedCall!^encodeAb/StatefulPartitionedCall!^encodeBa/StatefulPartitionedCall!^encodeBb/StatefulPartitionedCall!^encodeCa/StatefulPartitionedCall!^encodeCb/StatefulPartitionedCall!^encodeDa/StatefulPartitionedCall!^encodeDb/StatefulPartitionedCall!^encodeEa/StatefulPartitionedCall!^encodeEb/StatefulPartitionedCall*^spatial_dropout2d/StatefulPartitionedCall,^spatial_dropout2d_1/StatefulPartitionedCall#^transconvA/StatefulPartitionedCall#^transconvB/StatefulPartitionedCall#^transconvC/StatefulPartitionedCall#^transconvE/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes{
y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&PredictionMask/StatefulPartitionedCall&PredictionMask/StatefulPartitionedCall2D
 convOuta/StatefulPartitionedCall convOuta/StatefulPartitionedCall2D
 convOutb/StatefulPartitionedCall convOutb/StatefulPartitionedCall2D
 decodeAa/StatefulPartitionedCall decodeAa/StatefulPartitionedCall2D
 decodeAb/StatefulPartitionedCall decodeAb/StatefulPartitionedCall2D
 decodeBa/StatefulPartitionedCall decodeBa/StatefulPartitionedCall2D
 decodeBb/StatefulPartitionedCall decodeBb/StatefulPartitionedCall2D
 decodeCa/StatefulPartitionedCall decodeCa/StatefulPartitionedCall2D
 decodeCb/StatefulPartitionedCall decodeCb/StatefulPartitionedCall2D
 encodeAa/StatefulPartitionedCall encodeAa/StatefulPartitionedCall2D
 encodeAb/StatefulPartitionedCall encodeAb/StatefulPartitionedCall2D
 encodeBa/StatefulPartitionedCall encodeBa/StatefulPartitionedCall2D
 encodeBb/StatefulPartitionedCall encodeBb/StatefulPartitionedCall2D
 encodeCa/StatefulPartitionedCall encodeCa/StatefulPartitionedCall2D
 encodeCb/StatefulPartitionedCall encodeCb/StatefulPartitionedCall2D
 encodeDa/StatefulPartitionedCall encodeDa/StatefulPartitionedCall2D
 encodeDb/StatefulPartitionedCall encodeDb/StatefulPartitionedCall2D
 encodeEa/StatefulPartitionedCall encodeEa/StatefulPartitionedCall2D
 encodeEb/StatefulPartitionedCall encodeEb/StatefulPartitionedCall2V
)spatial_dropout2d/StatefulPartitionedCall)spatial_dropout2d/StatefulPartitionedCall2Z
+spatial_dropout2d_1/StatefulPartitionedCall+spatial_dropout2d_1/StatefulPartitionedCall2H
"transconvA/StatefulPartitionedCall"transconvA/StatefulPartitionedCall2H
"transconvB/StatefulPartitionedCall"transconvB/StatefulPartitionedCall2H
"transconvC/StatefulPartitionedCall"transconvC/StatefulPartitionedCall2H
"transconvE/StatefulPartitionedCall"transconvE/StatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
MRImages:&"
 
_user_specified_name191013:&"
 
_user_specified_name191015:&"
 
_user_specified_name191029:&"
 
_user_specified_name191031:&"
 
_user_specified_name191046:&"
 
_user_specified_name191048:&"
 
_user_specified_name191062:&"
 
_user_specified_name191064:&	"
 
_user_specified_name191079:&
"
 
_user_specified_name191081:&"
 
_user_specified_name191096:&"
 
_user_specified_name191098:&"
 
_user_specified_name191113:&"
 
_user_specified_name191115:&"
 
_user_specified_name191130:&"
 
_user_specified_name191132:&"
 
_user_specified_name191147:&"
 
_user_specified_name191149:&"
 
_user_specified_name191163:&"
 
_user_specified_name191165:&"
 
_user_specified_name191168:&"
 
_user_specified_name191170:&"
 
_user_specified_name191192:&"
 
_user_specified_name191194:&"
 
_user_specified_name191208:&"
 
_user_specified_name191210:&"
 
_user_specified_name191213:&"
 
_user_specified_name191215:&"
 
_user_specified_name191237:&"
 
_user_specified_name191239:&"
 
_user_specified_name191253:& "
 
_user_specified_name191255:&!"
 
_user_specified_name191258:&""
 
_user_specified_name191260:&#"
 
_user_specified_name191282:&$"
 
_user_specified_name191284:&%"
 
_user_specified_name191298:&&"
 
_user_specified_name191300:&'"
 
_user_specified_name191303:&("
 
_user_specified_name191305:&)"
 
_user_specified_name191327:&*"
 
_user_specified_name191329:&+"
 
_user_specified_name191343:&,"
 
_user_specified_name191345:&-"
 
_user_specified_name191359:&."
 
_user_specified_name191361
�
�
D__inference_encodeCb_layer_call_and_return_conditional_losses_192161

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_decodeAa_layer_call_and_return_conditional_losses_192564

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeDb_layer_call_and_return_conditional_losses_191129

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeBa_layer_call_and_return_conditional_losses_192053

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
J__inference_PredictionMask_layer_call_and_return_conditional_losses_192699

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
m
C__inference_concatB_layer_call_and_return_conditional_losses_191269

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:���������@@@_
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������@@ :���������@@ :W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�
�
D__inference_convOutb_layer_call_and_return_conditional_losses_192679

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
N
2__inference_spatial_dropout2d_layer_call_fn_192113

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_190763�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_encodeBb_layer_call_and_return_conditional_losses_192073

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_encodeBb_layer_call_and_return_conditional_losses_191061

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_encodeCa_layer_call_fn_192092

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeCa_layer_call_and_return_conditional_losses_191078w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs:&"
 
_user_specified_name192086:&"
 
_user_specified_name192088
�
l
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_190758

inputs
identity�I
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4������������������������������������`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"������������������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"������������������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*J
_output_shapes8
6:4�������������������������������������
IdentityIdentitydropout/SelectV2:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_decodeCa_layer_call_and_return_conditional_losses_191191

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_encodeDb_layer_call_fn_192238

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeDb_layer_call_and_return_conditional_losses_191129x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192232:&"
 
_user_specified_name192234
�
k
2__inference_spatial_dropout2d_layer_call_fn_192108

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_190758�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*J
_output_shapes8
6:4������������������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�!
�
F__inference_transconvE_layer_call_and_return_conditional_losses_190864

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_encodeCb_layer_call_fn_192150

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeCb_layer_call_and_return_conditional_losses_191095w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs:&"
 
_user_specified_name192144:&"
 
_user_specified_name192146
�
�
)__inference_encodeAb_layer_call_fn_192012

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_encodeAb_layer_call_and_return_conditional_losses_191028y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:&"
 
_user_specified_name192006:&"
 
_user_specified_name192008
�
P
4__inference_spatial_dropout2d_1_layer_call_fn_192201

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_190811�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_encodeAa_layer_call_and_return_conditional_losses_191012

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
+__inference_transconvB_layer_call_fn_192498

inputs!
unknown: @
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvB_layer_call_and_return_conditional_losses_190948�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:&"
 
_user_specified_name192492:&"
 
_user_specified_name192494
�
�
D__inference_encodeCa_layer_call_and_return_conditional_losses_192103

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�!
�
F__inference_transconvB_layer_call_and_return_conditional_losses_190948

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_decodeAb_layer_call_and_return_conditional_losses_191297

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_convOuta_layer_call_and_return_conditional_losses_192659

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
]
A__inference_poolC_layer_call_and_return_conditional_losses_192171

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�c
"__inference__traced_restore_194116
file_prefix:
 assignvariableop_encodeaa_kernel:.
 assignvariableop_1_encodeaa_bias:<
"assignvariableop_2_encodeab_kernel:.
 assignvariableop_3_encodeab_bias:<
"assignvariableop_4_encodeba_kernel: .
 assignvariableop_5_encodeba_bias: <
"assignvariableop_6_encodebb_kernel:  .
 assignvariableop_7_encodebb_bias: <
"assignvariableop_8_encodeca_kernel: @.
 assignvariableop_9_encodeca_bias:@=
#assignvariableop_10_encodecb_kernel:@@/
!assignvariableop_11_encodecb_bias:@>
#assignvariableop_12_encodeda_kernel:@�0
!assignvariableop_13_encodeda_bias:	�?
#assignvariableop_14_encodedb_kernel:��0
!assignvariableop_15_encodedb_bias:	�?
#assignvariableop_16_encodeea_kernel:��0
!assignvariableop_17_encodeea_bias:	�?
#assignvariableop_18_encodeeb_kernel:��0
!assignvariableop_19_encodeeb_bias:	�A
%assignvariableop_20_transconve_kernel:��2
#assignvariableop_21_transconve_bias:	�?
#assignvariableop_22_decodeca_kernel:��0
!assignvariableop_23_decodeca_bias:	�?
#assignvariableop_24_decodecb_kernel:��0
!assignvariableop_25_decodecb_bias:	�@
%assignvariableop_26_transconvc_kernel:@�1
#assignvariableop_27_transconvc_bias:@>
#assignvariableop_28_decodeba_kernel:�@/
!assignvariableop_29_decodeba_bias:@=
#assignvariableop_30_decodebb_kernel:@@/
!assignvariableop_31_decodebb_bias:@?
%assignvariableop_32_transconvb_kernel: @1
#assignvariableop_33_transconvb_bias: =
#assignvariableop_34_decodeaa_kernel:@ /
!assignvariableop_35_decodeaa_bias: =
#assignvariableop_36_decodeab_kernel:  /
!assignvariableop_37_decodeab_bias: ?
%assignvariableop_38_transconva_kernel: 1
#assignvariableop_39_transconva_bias:=
#assignvariableop_40_convouta_kernel: /
!assignvariableop_41_convouta_bias:=
#assignvariableop_42_convoutb_kernel:/
!assignvariableop_43_convoutb_bias:C
)assignvariableop_44_predictionmask_kernel:5
'assignvariableop_45_predictionmask_bias:'
assignvariableop_46_iteration:	 +
!assignvariableop_47_learning_rate: D
*assignvariableop_48_adam_m_encodeaa_kernel:D
*assignvariableop_49_adam_v_encodeaa_kernel:6
(assignvariableop_50_adam_m_encodeaa_bias:6
(assignvariableop_51_adam_v_encodeaa_bias:D
*assignvariableop_52_adam_m_encodeab_kernel:D
*assignvariableop_53_adam_v_encodeab_kernel:6
(assignvariableop_54_adam_m_encodeab_bias:6
(assignvariableop_55_adam_v_encodeab_bias:D
*assignvariableop_56_adam_m_encodeba_kernel: D
*assignvariableop_57_adam_v_encodeba_kernel: 6
(assignvariableop_58_adam_m_encodeba_bias: 6
(assignvariableop_59_adam_v_encodeba_bias: D
*assignvariableop_60_adam_m_encodebb_kernel:  D
*assignvariableop_61_adam_v_encodebb_kernel:  6
(assignvariableop_62_adam_m_encodebb_bias: 6
(assignvariableop_63_adam_v_encodebb_bias: D
*assignvariableop_64_adam_m_encodeca_kernel: @D
*assignvariableop_65_adam_v_encodeca_kernel: @6
(assignvariableop_66_adam_m_encodeca_bias:@6
(assignvariableop_67_adam_v_encodeca_bias:@D
*assignvariableop_68_adam_m_encodecb_kernel:@@D
*assignvariableop_69_adam_v_encodecb_kernel:@@6
(assignvariableop_70_adam_m_encodecb_bias:@6
(assignvariableop_71_adam_v_encodecb_bias:@E
*assignvariableop_72_adam_m_encodeda_kernel:@�E
*assignvariableop_73_adam_v_encodeda_kernel:@�7
(assignvariableop_74_adam_m_encodeda_bias:	�7
(assignvariableop_75_adam_v_encodeda_bias:	�F
*assignvariableop_76_adam_m_encodedb_kernel:��F
*assignvariableop_77_adam_v_encodedb_kernel:��7
(assignvariableop_78_adam_m_encodedb_bias:	�7
(assignvariableop_79_adam_v_encodedb_bias:	�F
*assignvariableop_80_adam_m_encodeea_kernel:��F
*assignvariableop_81_adam_v_encodeea_kernel:��7
(assignvariableop_82_adam_m_encodeea_bias:	�7
(assignvariableop_83_adam_v_encodeea_bias:	�F
*assignvariableop_84_adam_m_encodeeb_kernel:��F
*assignvariableop_85_adam_v_encodeeb_kernel:��7
(assignvariableop_86_adam_m_encodeeb_bias:	�7
(assignvariableop_87_adam_v_encodeeb_bias:	�H
,assignvariableop_88_adam_m_transconve_kernel:��H
,assignvariableop_89_adam_v_transconve_kernel:��9
*assignvariableop_90_adam_m_transconve_bias:	�9
*assignvariableop_91_adam_v_transconve_bias:	�F
*assignvariableop_92_adam_m_decodeca_kernel:��F
*assignvariableop_93_adam_v_decodeca_kernel:��7
(assignvariableop_94_adam_m_decodeca_bias:	�7
(assignvariableop_95_adam_v_decodeca_bias:	�F
*assignvariableop_96_adam_m_decodecb_kernel:��F
*assignvariableop_97_adam_v_decodecb_kernel:��7
(assignvariableop_98_adam_m_decodecb_bias:	�7
(assignvariableop_99_adam_v_decodecb_bias:	�H
-assignvariableop_100_adam_m_transconvc_kernel:@�H
-assignvariableop_101_adam_v_transconvc_kernel:@�9
+assignvariableop_102_adam_m_transconvc_bias:@9
+assignvariableop_103_adam_v_transconvc_bias:@F
+assignvariableop_104_adam_m_decodeba_kernel:�@F
+assignvariableop_105_adam_v_decodeba_kernel:�@7
)assignvariableop_106_adam_m_decodeba_bias:@7
)assignvariableop_107_adam_v_decodeba_bias:@E
+assignvariableop_108_adam_m_decodebb_kernel:@@E
+assignvariableop_109_adam_v_decodebb_kernel:@@7
)assignvariableop_110_adam_m_decodebb_bias:@7
)assignvariableop_111_adam_v_decodebb_bias:@G
-assignvariableop_112_adam_m_transconvb_kernel: @G
-assignvariableop_113_adam_v_transconvb_kernel: @9
+assignvariableop_114_adam_m_transconvb_bias: 9
+assignvariableop_115_adam_v_transconvb_bias: E
+assignvariableop_116_adam_m_decodeaa_kernel:@ E
+assignvariableop_117_adam_v_decodeaa_kernel:@ 7
)assignvariableop_118_adam_m_decodeaa_bias: 7
)assignvariableop_119_adam_v_decodeaa_bias: E
+assignvariableop_120_adam_m_decodeab_kernel:  E
+assignvariableop_121_adam_v_decodeab_kernel:  7
)assignvariableop_122_adam_m_decodeab_bias: 7
)assignvariableop_123_adam_v_decodeab_bias: G
-assignvariableop_124_adam_m_transconva_kernel: G
-assignvariableop_125_adam_v_transconva_kernel: 9
+assignvariableop_126_adam_m_transconva_bias:9
+assignvariableop_127_adam_v_transconva_bias:E
+assignvariableop_128_adam_m_convouta_kernel: E
+assignvariableop_129_adam_v_convouta_kernel: 7
)assignvariableop_130_adam_m_convouta_bias:7
)assignvariableop_131_adam_v_convouta_bias:E
+assignvariableop_132_adam_m_convoutb_kernel:E
+assignvariableop_133_adam_v_convoutb_kernel:7
)assignvariableop_134_adam_m_convoutb_bias:7
)assignvariableop_135_adam_v_convoutb_bias:K
1assignvariableop_136_adam_m_predictionmask_kernel:K
1assignvariableop_137_adam_v_predictionmask_kernel:=
/assignvariableop_138_adam_m_predictionmask_bias:=
/assignvariableop_139_adam_v_predictionmask_bias:&
assignvariableop_140_total_6: &
assignvariableop_141_count_6: &
assignvariableop_142_total_5: &
assignvariableop_143_count_5: &
assignvariableop_144_total_4: &
assignvariableop_145_count_4: &
assignvariableop_146_total_3: &
assignvariableop_147_count_3: &
assignvariableop_148_total_2: &
assignvariableop_149_count_2: &
assignvariableop_150_total_1: &
assignvariableop_151_count_1: $
assignvariableop_152_total: $
assignvariableop_153_count: 
identity_155��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_149�AssignVariableOp_15�AssignVariableOp_150�AssignVariableOp_151�AssignVariableOp_152�AssignVariableOp_153�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�A
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�@
value�@B�@�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/63/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/64/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/65/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/66/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/67/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/68/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/69/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/70/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/71/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/72/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/73/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/74/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/75/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/76/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/77/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/78/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/79/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/80/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/81/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/82/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/83/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/84/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/85/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/86/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/87/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/88/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/89/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/90/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/91/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/92/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_encodeaa_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_encodeaa_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_encodeab_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_encodeab_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_encodeba_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_encodeba_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_encodebb_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_encodebb_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_encodeca_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_encodeca_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_encodecb_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_encodecb_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_encodeda_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_encodeda_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_encodedb_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_encodedb_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_encodeea_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_encodeea_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_encodeeb_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_encodeeb_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_transconve_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp#assignvariableop_21_transconve_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_decodeca_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_decodeca_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_decodecb_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_decodecb_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_transconvc_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp#assignvariableop_27_transconvc_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_decodeba_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp!assignvariableop_29_decodeba_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp#assignvariableop_30_decodebb_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_decodebb_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp%assignvariableop_32_transconvb_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp#assignvariableop_33_transconvb_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_decodeaa_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_decodeaa_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_decodeab_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_decodeab_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp%assignvariableop_38_transconva_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp#assignvariableop_39_transconva_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_convouta_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_convouta_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp#assignvariableop_42_convoutb_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp!assignvariableop_43_convoutb_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_predictionmask_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_predictionmask_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpassignvariableop_46_iterationIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp!assignvariableop_47_learning_rateIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_encodeaa_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_encodeaa_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_encodeaa_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_encodeaa_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_m_encodeab_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_v_encodeab_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_m_encodeab_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_v_encodeab_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_m_encodeba_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_v_encodeba_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_m_encodeba_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_v_encodeba_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_m_encodebb_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_v_encodebb_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_m_encodebb_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_v_encodebb_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_m_encodeca_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_v_encodeca_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_m_encodeca_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_v_encodeca_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_m_encodecb_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_v_encodecb_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_m_encodecb_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_v_encodecb_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_m_encodeda_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_v_encodeda_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_m_encodeda_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_v_encodeda_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_m_encodedb_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_v_encodedb_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_m_encodedb_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_v_encodedb_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_m_encodeea_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_v_encodeea_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_m_encodeea_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_v_encodeea_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_m_encodeeb_kernelIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_v_encodeeb_kernelIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_m_encodeeb_biasIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_v_encodeeb_biasIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp,assignvariableop_88_adam_m_transconve_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_v_transconve_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp*assignvariableop_90_adam_m_transconve_biasIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_v_transconve_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp*assignvariableop_92_adam_m_decodeca_kernelIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_v_decodeca_kernelIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_m_decodeca_biasIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp(assignvariableop_95_adam_v_decodeca_biasIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_m_decodecb_kernelIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_v_decodecb_kernelIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_m_decodecb_biasIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_v_decodecb_biasIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp-assignvariableop_100_adam_m_transconvc_kernelIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp-assignvariableop_101_adam_v_transconvc_kernelIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp+assignvariableop_102_adam_m_transconvc_biasIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_v_transconvc_biasIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp+assignvariableop_104_adam_m_decodeba_kernelIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_v_decodeba_kernelIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_m_decodeba_biasIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp)assignvariableop_107_adam_v_decodeba_biasIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp+assignvariableop_108_adam_m_decodebb_kernelIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp+assignvariableop_109_adam_v_decodebb_kernelIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp)assignvariableop_110_adam_m_decodebb_biasIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp)assignvariableop_111_adam_v_decodebb_biasIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp-assignvariableop_112_adam_m_transconvb_kernelIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp-assignvariableop_113_adam_v_transconvb_kernelIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp+assignvariableop_114_adam_m_transconvb_biasIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp+assignvariableop_115_adam_v_transconvb_biasIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp+assignvariableop_116_adam_m_decodeaa_kernelIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp+assignvariableop_117_adam_v_decodeaa_kernelIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp)assignvariableop_118_adam_m_decodeaa_biasIdentity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp)assignvariableop_119_adam_v_decodeaa_biasIdentity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp+assignvariableop_120_adam_m_decodeab_kernelIdentity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp+assignvariableop_121_adam_v_decodeab_kernelIdentity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp)assignvariableop_122_adam_m_decodeab_biasIdentity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp)assignvariableop_123_adam_v_decodeab_biasIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp-assignvariableop_124_adam_m_transconva_kernelIdentity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp-assignvariableop_125_adam_v_transconva_kernelIdentity_125:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp+assignvariableop_126_adam_m_transconva_biasIdentity_126:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp+assignvariableop_127_adam_v_transconva_biasIdentity_127:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp+assignvariableop_128_adam_m_convouta_kernelIdentity_128:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp+assignvariableop_129_adam_v_convouta_kernelIdentity_129:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp)assignvariableop_130_adam_m_convouta_biasIdentity_130:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOp)assignvariableop_131_adam_v_convouta_biasIdentity_131:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOp+assignvariableop_132_adam_m_convoutb_kernelIdentity_132:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOp+assignvariableop_133_adam_v_convoutb_kernelIdentity_133:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOp)assignvariableop_134_adam_m_convoutb_biasIdentity_134:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOp)assignvariableop_135_adam_v_convoutb_biasIdentity_135:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_136AssignVariableOp1assignvariableop_136_adam_m_predictionmask_kernelIdentity_136:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_137AssignVariableOp1assignvariableop_137_adam_v_predictionmask_kernelIdentity_137:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_138AssignVariableOp/assignvariableop_138_adam_m_predictionmask_biasIdentity_138:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_139AssignVariableOp/assignvariableop_139_adam_v_predictionmask_biasIdentity_139:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_140AssignVariableOpassignvariableop_140_total_6Identity_140:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_141AssignVariableOpassignvariableop_141_count_6Identity_141:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_142AssignVariableOpassignvariableop_142_total_5Identity_142:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_143AssignVariableOpassignvariableop_143_count_5Identity_143:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_144AssignVariableOpassignvariableop_144_total_4Identity_144:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_145AssignVariableOpassignvariableop_145_count_4Identity_145:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_146AssignVariableOpassignvariableop_146_total_3Identity_146:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_147AssignVariableOpassignvariableop_147_count_3Identity_147:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_148AssignVariableOpassignvariableop_148_total_2Identity_148:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_149AssignVariableOpassignvariableop_149_count_2Identity_149:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_150AssignVariableOpassignvariableop_150_total_1Identity_150:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_151AssignVariableOpassignvariableop_151_count_1Identity_151:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_152AssignVariableOpassignvariableop_152_totalIdentity_152:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_153AssignVariableOpassignvariableop_153_countIdentity_153:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_154Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_155IdentityIdentity_154:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_155Identity_155:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameencodeAa/kernel:-)
'
_user_specified_nameencodeAa/bias:/+
)
_user_specified_nameencodeAb/kernel:-)
'
_user_specified_nameencodeAb/bias:/+
)
_user_specified_nameencodeBa/kernel:-)
'
_user_specified_nameencodeBa/bias:/+
)
_user_specified_nameencodeBb/kernel:-)
'
_user_specified_nameencodeBb/bias:/	+
)
_user_specified_nameencodeCa/kernel:-
)
'
_user_specified_nameencodeCa/bias:/+
)
_user_specified_nameencodeCb/kernel:-)
'
_user_specified_nameencodeCb/bias:/+
)
_user_specified_nameencodeDa/kernel:-)
'
_user_specified_nameencodeDa/bias:/+
)
_user_specified_nameencodeDb/kernel:-)
'
_user_specified_nameencodeDb/bias:/+
)
_user_specified_nameencodeEa/kernel:-)
'
_user_specified_nameencodeEa/bias:/+
)
_user_specified_nameencodeEb/kernel:-)
'
_user_specified_nameencodeEb/bias:1-
+
_user_specified_nametransconvE/kernel:/+
)
_user_specified_nametransconvE/bias:/+
)
_user_specified_namedecodeCa/kernel:-)
'
_user_specified_namedecodeCa/bias:/+
)
_user_specified_namedecodeCb/kernel:-)
'
_user_specified_namedecodeCb/bias:1-
+
_user_specified_nametransconvC/kernel:/+
)
_user_specified_nametransconvC/bias:/+
)
_user_specified_namedecodeBa/kernel:-)
'
_user_specified_namedecodeBa/bias:/+
)
_user_specified_namedecodeBb/kernel:- )
'
_user_specified_namedecodeBb/bias:1!-
+
_user_specified_nametransconvB/kernel:/"+
)
_user_specified_nametransconvB/bias:/#+
)
_user_specified_namedecodeAa/kernel:-$)
'
_user_specified_namedecodeAa/bias:/%+
)
_user_specified_namedecodeAb/kernel:-&)
'
_user_specified_namedecodeAb/bias:1'-
+
_user_specified_nametransconvA/kernel:/(+
)
_user_specified_nametransconvA/bias:/)+
)
_user_specified_nameconvOuta/kernel:-*)
'
_user_specified_nameconvOuta/bias:/++
)
_user_specified_nameconvOutb/kernel:-,)
'
_user_specified_nameconvOutb/bias:5-1
/
_user_specified_namePredictionMask/kernel:3./
-
_user_specified_namePredictionMask/bias:)/%
#
_user_specified_name	iteration:-0)
'
_user_specified_namelearning_rate:612
0
_user_specified_nameAdam/m/encodeAa/kernel:622
0
_user_specified_nameAdam/v/encodeAa/kernel:430
.
_user_specified_nameAdam/m/encodeAa/bias:440
.
_user_specified_nameAdam/v/encodeAa/bias:652
0
_user_specified_nameAdam/m/encodeAb/kernel:662
0
_user_specified_nameAdam/v/encodeAb/kernel:470
.
_user_specified_nameAdam/m/encodeAb/bias:480
.
_user_specified_nameAdam/v/encodeAb/bias:692
0
_user_specified_nameAdam/m/encodeBa/kernel:6:2
0
_user_specified_nameAdam/v/encodeBa/kernel:4;0
.
_user_specified_nameAdam/m/encodeBa/bias:4<0
.
_user_specified_nameAdam/v/encodeBa/bias:6=2
0
_user_specified_nameAdam/m/encodeBb/kernel:6>2
0
_user_specified_nameAdam/v/encodeBb/kernel:4?0
.
_user_specified_nameAdam/m/encodeBb/bias:4@0
.
_user_specified_nameAdam/v/encodeBb/bias:6A2
0
_user_specified_nameAdam/m/encodeCa/kernel:6B2
0
_user_specified_nameAdam/v/encodeCa/kernel:4C0
.
_user_specified_nameAdam/m/encodeCa/bias:4D0
.
_user_specified_nameAdam/v/encodeCa/bias:6E2
0
_user_specified_nameAdam/m/encodeCb/kernel:6F2
0
_user_specified_nameAdam/v/encodeCb/kernel:4G0
.
_user_specified_nameAdam/m/encodeCb/bias:4H0
.
_user_specified_nameAdam/v/encodeCb/bias:6I2
0
_user_specified_nameAdam/m/encodeDa/kernel:6J2
0
_user_specified_nameAdam/v/encodeDa/kernel:4K0
.
_user_specified_nameAdam/m/encodeDa/bias:4L0
.
_user_specified_nameAdam/v/encodeDa/bias:6M2
0
_user_specified_nameAdam/m/encodeDb/kernel:6N2
0
_user_specified_nameAdam/v/encodeDb/kernel:4O0
.
_user_specified_nameAdam/m/encodeDb/bias:4P0
.
_user_specified_nameAdam/v/encodeDb/bias:6Q2
0
_user_specified_nameAdam/m/encodeEa/kernel:6R2
0
_user_specified_nameAdam/v/encodeEa/kernel:4S0
.
_user_specified_nameAdam/m/encodeEa/bias:4T0
.
_user_specified_nameAdam/v/encodeEa/bias:6U2
0
_user_specified_nameAdam/m/encodeEb/kernel:6V2
0
_user_specified_nameAdam/v/encodeEb/kernel:4W0
.
_user_specified_nameAdam/m/encodeEb/bias:4X0
.
_user_specified_nameAdam/v/encodeEb/bias:8Y4
2
_user_specified_nameAdam/m/transconvE/kernel:8Z4
2
_user_specified_nameAdam/v/transconvE/kernel:6[2
0
_user_specified_nameAdam/m/transconvE/bias:6\2
0
_user_specified_nameAdam/v/transconvE/bias:6]2
0
_user_specified_nameAdam/m/decodeCa/kernel:6^2
0
_user_specified_nameAdam/v/decodeCa/kernel:4_0
.
_user_specified_nameAdam/m/decodeCa/bias:4`0
.
_user_specified_nameAdam/v/decodeCa/bias:6a2
0
_user_specified_nameAdam/m/decodeCb/kernel:6b2
0
_user_specified_nameAdam/v/decodeCb/kernel:4c0
.
_user_specified_nameAdam/m/decodeCb/bias:4d0
.
_user_specified_nameAdam/v/decodeCb/bias:8e4
2
_user_specified_nameAdam/m/transconvC/kernel:8f4
2
_user_specified_nameAdam/v/transconvC/kernel:6g2
0
_user_specified_nameAdam/m/transconvC/bias:6h2
0
_user_specified_nameAdam/v/transconvC/bias:6i2
0
_user_specified_nameAdam/m/decodeBa/kernel:6j2
0
_user_specified_nameAdam/v/decodeBa/kernel:4k0
.
_user_specified_nameAdam/m/decodeBa/bias:4l0
.
_user_specified_nameAdam/v/decodeBa/bias:6m2
0
_user_specified_nameAdam/m/decodeBb/kernel:6n2
0
_user_specified_nameAdam/v/decodeBb/kernel:4o0
.
_user_specified_nameAdam/m/decodeBb/bias:4p0
.
_user_specified_nameAdam/v/decodeBb/bias:8q4
2
_user_specified_nameAdam/m/transconvB/kernel:8r4
2
_user_specified_nameAdam/v/transconvB/kernel:6s2
0
_user_specified_nameAdam/m/transconvB/bias:6t2
0
_user_specified_nameAdam/v/transconvB/bias:6u2
0
_user_specified_nameAdam/m/decodeAa/kernel:6v2
0
_user_specified_nameAdam/v/decodeAa/kernel:4w0
.
_user_specified_nameAdam/m/decodeAa/bias:4x0
.
_user_specified_nameAdam/v/decodeAa/bias:6y2
0
_user_specified_nameAdam/m/decodeAb/kernel:6z2
0
_user_specified_nameAdam/v/decodeAb/kernel:4{0
.
_user_specified_nameAdam/m/decodeAb/bias:4|0
.
_user_specified_nameAdam/v/decodeAb/bias:8}4
2
_user_specified_nameAdam/m/transconvA/kernel:8~4
2
_user_specified_nameAdam/v/transconvA/kernel:62
0
_user_specified_nameAdam/m/transconvA/bias:7�2
0
_user_specified_nameAdam/v/transconvA/bias:7�2
0
_user_specified_nameAdam/m/convOuta/kernel:7�2
0
_user_specified_nameAdam/v/convOuta/kernel:5�0
.
_user_specified_nameAdam/m/convOuta/bias:5�0
.
_user_specified_nameAdam/v/convOuta/bias:7�2
0
_user_specified_nameAdam/m/convOutb/kernel:7�2
0
_user_specified_nameAdam/v/convOutb/kernel:5�0
.
_user_specified_nameAdam/m/convOutb/bias:5�0
.
_user_specified_nameAdam/v/convOutb/bias:=�8
6
_user_specified_nameAdam/m/PredictionMask/kernel:=�8
6
_user_specified_nameAdam/v/PredictionMask/kernel:;�6
4
_user_specified_nameAdam/m/PredictionMask/bias:;�6
4
_user_specified_nameAdam/v/PredictionMask/bias:(�#
!
_user_specified_name	total_6:(�#
!
_user_specified_name	count_6:(�#
!
_user_specified_name	total_5:(�#
!
_user_specified_name	count_5:(�#
!
_user_specified_name	total_4:(�#
!
_user_specified_name	count_4:(�#
!
_user_specified_name	total_3:(�#
!
_user_specified_name	count_3:(�#
!
_user_specified_name	total_2:(�#
!
_user_specified_name	count_2:(�#
!
_user_specified_name	total_1:(�#
!
_user_specified_name	count_1:&�!

_user_specified_nametotal:&�!

_user_specified_namecount
�
�
+__inference_transconvC_layer_call_fn_192403

inputs"
unknown:@�
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_transconvC_layer_call_and_return_conditional_losses_190906�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:&"
 
_user_specified_name192397:&"
 
_user_specified_name192399"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
MRImages;
serving_default_MRImages:0�����������L
PredictionMask:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
�

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!layer_with_weights-21
!layer-32
"layer_with_weights-22
"layer-33
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_default_save_signature
*	optimizer
+
signatures"
_tf_keras_network
6
,_init_input_shape"
_tf_keras_input_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias
 V_jit_compiled_convolution_op"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias
 e_jit_compiled_convolution_op"
_tf_keras_layer
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l_random_generator"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
30
41
<2
=3
K4
L5
T6
U7
c8
d9
s10
t11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45"
trackable_list_wrapper
�
30
41
<2
=3
K4
L5
T6
U7
c8
d9
s10
t11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
)_default_save_signature
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191591
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191688�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191365
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191494�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_190715MRImages"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeAa_layer_call_fn_191992�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeAa_layer_call_and_return_conditional_losses_192003�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'2encodeAa/kernel
:2encodeAa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeAb_layer_call_fn_192012�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeAb_layer_call_and_return_conditional_losses_192023�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'2encodeAb/kernel
:2encodeAb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_poolA_layer_call_fn_192028�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_poolA_layer_call_and_return_conditional_losses_192033�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeBa_layer_call_fn_192042�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeBa_layer_call_and_return_conditional_losses_192053�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):' 2encodeBa/kernel
: 2encodeBa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeBb_layer_call_fn_192062�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeBb_layer_call_and_return_conditional_losses_192073�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'  2encodeBb/kernel
: 2encodeBb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_poolB_layer_call_fn_192078�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_poolB_layer_call_and_return_conditional_losses_192083�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeCa_layer_call_fn_192092�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeCa_layer_call_and_return_conditional_losses_192103�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):' @2encodeCa/kernel
:@2encodeCa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_spatial_dropout2d_layer_call_fn_192108
2__inference_spatial_dropout2d_layer_call_fn_192113�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192136
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192141�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeCb_layer_call_fn_192150�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeCb_layer_call_and_return_conditional_losses_192161�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'@@2encodeCb/kernel
:@2encodeCb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_poolC_layer_call_fn_192166�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_poolC_layer_call_and_return_conditional_losses_192171�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeDa_layer_call_fn_192180�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeDa_layer_call_and_return_conditional_losses_192191�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(@�2encodeDa/kernel
:�2encodeDa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_spatial_dropout2d_1_layer_call_fn_192196
4__inference_spatial_dropout2d_1_layer_call_fn_192201�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192224
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192229�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeDb_layer_call_fn_192238�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeDb_layer_call_and_return_conditional_losses_192249�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2encodeDb/kernel
:�2encodeDb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_poolD_layer_call_fn_192254�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_poolD_layer_call_and_return_conditional_losses_192259�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeEa_layer_call_fn_192268�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeEa_layer_call_and_return_conditional_losses_192279�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2encodeEa/kernel
:�2encodeEa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_encodeEb_layer_call_fn_192288�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_encodeEb_layer_call_and_return_conditional_losses_192299�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2encodeEb/kernel
:�2encodeEb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_transconvE_layer_call_fn_192308�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_transconvE_layer_call_and_return_conditional_losses_192341�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+��2transconvE/kernel
:�2transconvE/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_concatD_layer_call_fn_192347�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_concatD_layer_call_and_return_conditional_losses_192354�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_decodeCa_layer_call_fn_192363�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_decodeCa_layer_call_and_return_conditional_losses_192374�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2decodeCa/kernel
:�2decodeCa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_decodeCb_layer_call_fn_192383�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_decodeCb_layer_call_and_return_conditional_losses_192394�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)��2decodeCb/kernel
:�2decodeCb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_transconvC_layer_call_fn_192403�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_transconvC_layer_call_and_return_conditional_losses_192436�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
,:*@�2transconvC/kernel
:@2transconvC/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_concatC_layer_call_fn_192442�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_concatC_layer_call_and_return_conditional_losses_192449�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_decodeBa_layer_call_fn_192458�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_decodeBa_layer_call_and_return_conditional_losses_192469�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(�@2decodeBa/kernel
:@2decodeBa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_decodeBb_layer_call_fn_192478�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_decodeBb_layer_call_and_return_conditional_losses_192489�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'@@2decodeBb/kernel
:@2decodeBb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_transconvB_layer_call_fn_192498�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_transconvB_layer_call_and_return_conditional_losses_192531�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:) @2transconvB/kernel
: 2transconvB/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_concatB_layer_call_fn_192537�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_concatB_layer_call_and_return_conditional_losses_192544�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_decodeAa_layer_call_fn_192553�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_decodeAa_layer_call_and_return_conditional_losses_192564�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'@ 2decodeAa/kernel
: 2decodeAa/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_decodeAb_layer_call_fn_192573�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_decodeAb_layer_call_and_return_conditional_losses_192584�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'  2decodeAb/kernel
: 2decodeAb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_transconvA_layer_call_fn_192593�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_transconvA_layer_call_and_return_conditional_losses_192626�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:) 2transconvA/kernel
:2transconvA/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_concatA_layer_call_fn_192632�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_concatA_layer_call_and_return_conditional_losses_192639�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_convOuta_layer_call_fn_192648�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_convOuta_layer_call_and_return_conditional_losses_192659�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):' 2convOuta/kernel
:2convOuta/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_convOutb_layer_call_fn_192668�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_convOutb_layer_call_and_return_conditional_losses_192679�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
):'2convOutb/kernel
:2convOutb/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_PredictionMask_layer_call_fn_192688�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_PredictionMask_layer_call_and_return_conditional_losses_192699�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:-2PredictionMask/kernel
!:2PredictionMask/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191591MRImages"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191688MRImages"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191365MRImages"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191494MRImages"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89
�90
�91
�92"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_191983MRImages"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jMRImages
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeAa_layer_call_fn_191992inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeAa_layer_call_and_return_conditional_losses_192003inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeAb_layer_call_fn_192012inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeAb_layer_call_and_return_conditional_losses_192023inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_poolA_layer_call_fn_192028inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_poolA_layer_call_and_return_conditional_losses_192033inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeBa_layer_call_fn_192042inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeBa_layer_call_and_return_conditional_losses_192053inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeBb_layer_call_fn_192062inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeBb_layer_call_and_return_conditional_losses_192073inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_poolB_layer_call_fn_192078inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_poolB_layer_call_and_return_conditional_losses_192083inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeCa_layer_call_fn_192092inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeCa_layer_call_and_return_conditional_losses_192103inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_spatial_dropout2d_layer_call_fn_192108inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_spatial_dropout2d_layer_call_fn_192113inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192136inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192141inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeCb_layer_call_fn_192150inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeCb_layer_call_and_return_conditional_losses_192161inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_poolC_layer_call_fn_192166inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_poolC_layer_call_and_return_conditional_losses_192171inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeDa_layer_call_fn_192180inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeDa_layer_call_and_return_conditional_losses_192191inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_spatial_dropout2d_1_layer_call_fn_192196inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_spatial_dropout2d_1_layer_call_fn_192201inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192224inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192229inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeDb_layer_call_fn_192238inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeDb_layer_call_and_return_conditional_losses_192249inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_poolD_layer_call_fn_192254inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_poolD_layer_call_and_return_conditional_losses_192259inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeEa_layer_call_fn_192268inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeEa_layer_call_and_return_conditional_losses_192279inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_encodeEb_layer_call_fn_192288inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_encodeEb_layer_call_and_return_conditional_losses_192299inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_transconvE_layer_call_fn_192308inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_transconvE_layer_call_and_return_conditional_losses_192341inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_concatD_layer_call_fn_192347inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_concatD_layer_call_and_return_conditional_losses_192354inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decodeCa_layer_call_fn_192363inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_decodeCa_layer_call_and_return_conditional_losses_192374inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decodeCb_layer_call_fn_192383inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_decodeCb_layer_call_and_return_conditional_losses_192394inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_transconvC_layer_call_fn_192403inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_transconvC_layer_call_and_return_conditional_losses_192436inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_concatC_layer_call_fn_192442inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_concatC_layer_call_and_return_conditional_losses_192449inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decodeBa_layer_call_fn_192458inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_decodeBa_layer_call_and_return_conditional_losses_192469inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decodeBb_layer_call_fn_192478inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_decodeBb_layer_call_and_return_conditional_losses_192489inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_transconvB_layer_call_fn_192498inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_transconvB_layer_call_and_return_conditional_losses_192531inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_concatB_layer_call_fn_192537inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_concatB_layer_call_and_return_conditional_losses_192544inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decodeAa_layer_call_fn_192553inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_decodeAa_layer_call_and_return_conditional_losses_192564inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decodeAb_layer_call_fn_192573inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_decodeAb_layer_call_and_return_conditional_losses_192584inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_transconvA_layer_call_fn_192593inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_transconvA_layer_call_and_return_conditional_losses_192626inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_concatA_layer_call_fn_192632inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_concatA_layer_call_and_return_conditional_losses_192639inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_convOuta_layer_call_fn_192648inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_convOuta_layer_call_and_return_conditional_losses_192659inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_convOutb_layer_call_fn_192668inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_convOutb_layer_call_and_return_conditional_losses_192679inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_PredictionMask_layer_call_fn_192688inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_PredictionMask_layer_call_and_return_conditional_losses_192699inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
.:,2Adam/m/encodeAa/kernel
.:,2Adam/v/encodeAa/kernel
 :2Adam/m/encodeAa/bias
 :2Adam/v/encodeAa/bias
.:,2Adam/m/encodeAb/kernel
.:,2Adam/v/encodeAb/kernel
 :2Adam/m/encodeAb/bias
 :2Adam/v/encodeAb/bias
.:, 2Adam/m/encodeBa/kernel
.:, 2Adam/v/encodeBa/kernel
 : 2Adam/m/encodeBa/bias
 : 2Adam/v/encodeBa/bias
.:,  2Adam/m/encodeBb/kernel
.:,  2Adam/v/encodeBb/kernel
 : 2Adam/m/encodeBb/bias
 : 2Adam/v/encodeBb/bias
.:, @2Adam/m/encodeCa/kernel
.:, @2Adam/v/encodeCa/kernel
 :@2Adam/m/encodeCa/bias
 :@2Adam/v/encodeCa/bias
.:,@@2Adam/m/encodeCb/kernel
.:,@@2Adam/v/encodeCb/kernel
 :@2Adam/m/encodeCb/bias
 :@2Adam/v/encodeCb/bias
/:-@�2Adam/m/encodeDa/kernel
/:-@�2Adam/v/encodeDa/kernel
!:�2Adam/m/encodeDa/bias
!:�2Adam/v/encodeDa/bias
0:.��2Adam/m/encodeDb/kernel
0:.��2Adam/v/encodeDb/kernel
!:�2Adam/m/encodeDb/bias
!:�2Adam/v/encodeDb/bias
0:.��2Adam/m/encodeEa/kernel
0:.��2Adam/v/encodeEa/kernel
!:�2Adam/m/encodeEa/bias
!:�2Adam/v/encodeEa/bias
0:.��2Adam/m/encodeEb/kernel
0:.��2Adam/v/encodeEb/kernel
!:�2Adam/m/encodeEb/bias
!:�2Adam/v/encodeEb/bias
2:0��2Adam/m/transconvE/kernel
2:0��2Adam/v/transconvE/kernel
#:!�2Adam/m/transconvE/bias
#:!�2Adam/v/transconvE/bias
0:.��2Adam/m/decodeCa/kernel
0:.��2Adam/v/decodeCa/kernel
!:�2Adam/m/decodeCa/bias
!:�2Adam/v/decodeCa/bias
0:.��2Adam/m/decodeCb/kernel
0:.��2Adam/v/decodeCb/kernel
!:�2Adam/m/decodeCb/bias
!:�2Adam/v/decodeCb/bias
1:/@�2Adam/m/transconvC/kernel
1:/@�2Adam/v/transconvC/kernel
": @2Adam/m/transconvC/bias
": @2Adam/v/transconvC/bias
/:-�@2Adam/m/decodeBa/kernel
/:-�@2Adam/v/decodeBa/kernel
 :@2Adam/m/decodeBa/bias
 :@2Adam/v/decodeBa/bias
.:,@@2Adam/m/decodeBb/kernel
.:,@@2Adam/v/decodeBb/kernel
 :@2Adam/m/decodeBb/bias
 :@2Adam/v/decodeBb/bias
0:. @2Adam/m/transconvB/kernel
0:. @2Adam/v/transconvB/kernel
":  2Adam/m/transconvB/bias
":  2Adam/v/transconvB/bias
.:,@ 2Adam/m/decodeAa/kernel
.:,@ 2Adam/v/decodeAa/kernel
 : 2Adam/m/decodeAa/bias
 : 2Adam/v/decodeAa/bias
.:,  2Adam/m/decodeAb/kernel
.:,  2Adam/v/decodeAb/kernel
 : 2Adam/m/decodeAb/bias
 : 2Adam/v/decodeAb/bias
0:. 2Adam/m/transconvA/kernel
0:. 2Adam/v/transconvA/kernel
": 2Adam/m/transconvA/bias
": 2Adam/v/transconvA/bias
.:, 2Adam/m/convOuta/kernel
.:, 2Adam/v/convOuta/kernel
 :2Adam/m/convOuta/bias
 :2Adam/v/convOuta/bias
.:,2Adam/m/convOutb/kernel
.:,2Adam/v/convOutb/kernel
 :2Adam/m/convOutb/bias
 :2Adam/v/convOutb/bias
4:22Adam/m/PredictionMask/kernel
4:22Adam/v/PredictionMask/kernel
&:$2Adam/m/PredictionMask/bias
&:$2Adam/v/PredictionMask/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191365�P34<=KLTUcdst����������������������������������C�@
9�6
,�)
MRImages�����������
p

 
� "6�3
,�)
tensor_0�����������
� �
R__inference_2DUNet_Brats_Decathlon_layer_call_and_return_conditional_losses_191494�P34<=KLTUcdst����������������������������������C�@
9�6
,�)
MRImages�����������
p 

 
� "6�3
,�)
tensor_0�����������
� �
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191591�P34<=KLTUcdst����������������������������������C�@
9�6
,�)
MRImages�����������
p

 
� "+�(
unknown������������
7__inference_2DUNet_Brats_Decathlon_layer_call_fn_191688�P34<=KLTUcdst����������������������������������C�@
9�6
,�)
MRImages�����������
p 

 
� "+�(
unknown������������
J__inference_PredictionMask_layer_call_and_return_conditional_losses_192699y��9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
/__inference_PredictionMask_layer_call_fn_192688n��9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
!__inference__wrapped_model_190715�P34<=KLTUcdst����������������������������������;�8
1�.
,�)
MRImages�����������
� "I�F
D
PredictionMask2�/
predictionmask������������
C__inference_concatA_layer_call_and_return_conditional_losses_192639�n�k
d�a
_�\
,�)
inputs_0�����������
,�)
inputs_1�����������
� "6�3
,�)
tensor_0����������� 
� �
(__inference_concatA_layer_call_fn_192632�n�k
d�a
_�\
,�)
inputs_0�����������
,�)
inputs_1�����������
� "+�(
unknown����������� �
C__inference_concatB_layer_call_and_return_conditional_losses_192544�j�g
`�]
[�X
*�'
inputs_0���������@@ 
*�'
inputs_1���������@@ 
� "4�1
*�'
tensor_0���������@@@
� �
(__inference_concatB_layer_call_fn_192537�j�g
`�]
[�X
*�'
inputs_0���������@@ 
*�'
inputs_1���������@@ 
� ")�&
unknown���������@@@�
C__inference_concatC_layer_call_and_return_conditional_losses_192449�j�g
`�]
[�X
*�'
inputs_0���������  @
*�'
inputs_1���������  @
� "5�2
+�(
tensor_0���������  �
� �
(__inference_concatC_layer_call_fn_192442�j�g
`�]
[�X
*�'
inputs_0���������  @
*�'
inputs_1���������  @
� "*�'
unknown���������  ��
C__inference_concatD_layer_call_and_return_conditional_losses_192354�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "5�2
+�(
tensor_0����������
� �
(__inference_concatD_layer_call_fn_192347�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "*�'
unknown�����������
D__inference_convOuta_layer_call_and_return_conditional_losses_192659y��9�6
/�,
*�'
inputs����������� 
� "6�3
,�)
tensor_0�����������
� �
)__inference_convOuta_layer_call_fn_192648n��9�6
/�,
*�'
inputs����������� 
� "+�(
unknown������������
D__inference_convOutb_layer_call_and_return_conditional_losses_192679y��9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
)__inference_convOutb_layer_call_fn_192668n��9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
D__inference_decodeAa_layer_call_and_return_conditional_losses_192564u��7�4
-�*
(�%
inputs���������@@@
� "4�1
*�'
tensor_0���������@@ 
� �
)__inference_decodeAa_layer_call_fn_192553j��7�4
-�*
(�%
inputs���������@@@
� ")�&
unknown���������@@ �
D__inference_decodeAb_layer_call_and_return_conditional_losses_192584u��7�4
-�*
(�%
inputs���������@@ 
� "4�1
*�'
tensor_0���������@@ 
� �
)__inference_decodeAb_layer_call_fn_192573j��7�4
-�*
(�%
inputs���������@@ 
� ")�&
unknown���������@@ �
D__inference_decodeBa_layer_call_and_return_conditional_losses_192469v��8�5
.�+
)�&
inputs���������  �
� "4�1
*�'
tensor_0���������  @
� �
)__inference_decodeBa_layer_call_fn_192458k��8�5
.�+
)�&
inputs���������  �
� ")�&
unknown���������  @�
D__inference_decodeBb_layer_call_and_return_conditional_losses_192489u��7�4
-�*
(�%
inputs���������  @
� "4�1
*�'
tensor_0���������  @
� �
)__inference_decodeBb_layer_call_fn_192478j��7�4
-�*
(�%
inputs���������  @
� ")�&
unknown���������  @�
D__inference_decodeCa_layer_call_and_return_conditional_losses_192374w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
)__inference_decodeCa_layer_call_fn_192363l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
D__inference_decodeCb_layer_call_and_return_conditional_losses_192394w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
)__inference_decodeCb_layer_call_fn_192383l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
D__inference_encodeAa_layer_call_and_return_conditional_losses_192003w349�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
)__inference_encodeAa_layer_call_fn_191992l349�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
D__inference_encodeAb_layer_call_and_return_conditional_losses_192023w<=9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
)__inference_encodeAb_layer_call_fn_192012l<=9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
D__inference_encodeBa_layer_call_and_return_conditional_losses_192053sKL7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@ 
� �
)__inference_encodeBa_layer_call_fn_192042hKL7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@ �
D__inference_encodeBb_layer_call_and_return_conditional_losses_192073sTU7�4
-�*
(�%
inputs���������@@ 
� "4�1
*�'
tensor_0���������@@ 
� �
)__inference_encodeBb_layer_call_fn_192062hTU7�4
-�*
(�%
inputs���������@@ 
� ")�&
unknown���������@@ �
D__inference_encodeCa_layer_call_and_return_conditional_losses_192103scd7�4
-�*
(�%
inputs���������   
� "4�1
*�'
tensor_0���������  @
� �
)__inference_encodeCa_layer_call_fn_192092hcd7�4
-�*
(�%
inputs���������   
� ")�&
unknown���������  @�
D__inference_encodeCb_layer_call_and_return_conditional_losses_192161sst7�4
-�*
(�%
inputs���������  @
� "4�1
*�'
tensor_0���������  @
� �
)__inference_encodeCb_layer_call_fn_192150hst7�4
-�*
(�%
inputs���������  @
� ")�&
unknown���������  @�
D__inference_encodeDa_layer_call_and_return_conditional_losses_192191v��7�4
-�*
(�%
inputs���������@
� "5�2
+�(
tensor_0����������
� �
)__inference_encodeDa_layer_call_fn_192180k��7�4
-�*
(�%
inputs���������@
� "*�'
unknown�����������
D__inference_encodeDb_layer_call_and_return_conditional_losses_192249w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
)__inference_encodeDb_layer_call_fn_192238l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
D__inference_encodeEa_layer_call_and_return_conditional_losses_192279w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
)__inference_encodeEa_layer_call_fn_192268l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
D__inference_encodeEb_layer_call_and_return_conditional_losses_192299w��8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
)__inference_encodeEb_layer_call_fn_192288l��8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
A__inference_poolA_layer_call_and_return_conditional_losses_192033�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
&__inference_poolA_layer_call_fn_192028�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
A__inference_poolB_layer_call_and_return_conditional_losses_192083�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
&__inference_poolB_layer_call_fn_192078�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
A__inference_poolC_layer_call_and_return_conditional_losses_192171�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
&__inference_poolC_layer_call_fn_192166�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
A__inference_poolD_layer_call_and_return_conditional_losses_192259�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
&__inference_poolD_layer_call_fn_192254�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
$__inference_signature_wrapper_191983�P34<=KLTUcdst����������������������������������G�D
� 
=�:
8
MRImages,�)
mrimages�����������"I�F
D
PredictionMask2�/
predictionmask������������
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192224�V�S
L�I
C�@
inputs4������������������������������������
p
� "O�L
E�B
tensor_04������������������������������������
� �
O__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_192229�V�S
L�I
C�@
inputs4������������������������������������
p 
� "O�L
E�B
tensor_04������������������������������������
� �
4__inference_spatial_dropout2d_1_layer_call_fn_192196�V�S
L�I
C�@
inputs4������������������������������������
p
� "D�A
unknown4�������������������������������������
4__inference_spatial_dropout2d_1_layer_call_fn_192201�V�S
L�I
C�@
inputs4������������������������������������
p 
� "D�A
unknown4�������������������������������������
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192136�V�S
L�I
C�@
inputs4������������������������������������
p
� "O�L
E�B
tensor_04������������������������������������
� �
M__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_192141�V�S
L�I
C�@
inputs4������������������������������������
p 
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_spatial_dropout2d_layer_call_fn_192108�V�S
L�I
C�@
inputs4������������������������������������
p
� "D�A
unknown4�������������������������������������
2__inference_spatial_dropout2d_layer_call_fn_192113�V�S
L�I
C�@
inputs4������������������������������������
p 
� "D�A
unknown4�������������������������������������
F__inference_transconvA_layer_call_and_return_conditional_losses_192626���I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������
� �
+__inference_transconvA_layer_call_fn_192593���I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+����������������������������
F__inference_transconvB_layer_call_and_return_conditional_losses_192531���I�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+��������������������������� 
� �
+__inference_transconvB_layer_call_fn_192498���I�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+��������������������������� �
F__inference_transconvC_layer_call_and_return_conditional_losses_192436���J�G
@�=
;�8
inputs,����������������������������
� "F�C
<�9
tensor_0+���������������������������@
� �
+__inference_transconvC_layer_call_fn_192403���J�G
@�=
;�8
inputs,����������������������������
� ";�8
unknown+���������������������������@�
F__inference_transconvE_layer_call_and_return_conditional_losses_192341���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
+__inference_transconvE_layer_call_fn_192308���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,����������������������������