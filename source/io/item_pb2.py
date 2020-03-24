# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: source/io/item.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='source/io/item.proto',
  package='music_io',
  syntax='proto2',
  serialized_pb=_b('\n\x14source/io/item.proto\x12\x08music_io\"~\n\x04Item\x12\"\n\x08\x66\x65\x61tures\x18\x01 \x02(\x0b\x32\x10.music_io.Matrix\x12\x12\n\nitem_class\x18\x02 \x01(\t\x12\x13\n\x0bitem_review\x18\x03 \x01(\t\x12\x14\n\x0citem_targets\x18\x04 \x03(\x01\x12\x13\n\x0btoken_class\x18\x05 \x03(\t\"3\n\x06Matrix\x12\x14\n\x0c\x66\x65\x61ture_size\x18\x01 \x02(\x05\x12\x13\n\x0b\x66lat_matrix\x18\x02 \x03(\x01')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_ITEM = _descriptor.Descriptor(
  name='Item',
  full_name='music_io.Item',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='music_io.Item.features', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='item_class', full_name='music_io.Item.item_class', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='item_review', full_name='music_io.Item.item_review', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='item_targets', full_name='music_io.Item.item_targets', index=3,
      number=4, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='token_class', full_name='music_io.Item.token_class', index=4,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=160,
)


_MATRIX = _descriptor.Descriptor(
  name='Matrix',
  full_name='music_io.Matrix',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_size', full_name='music_io.Matrix.feature_size', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='flat_matrix', full_name='music_io.Matrix.flat_matrix', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=162,
  serialized_end=213,
)

_ITEM.fields_by_name['features'].message_type = _MATRIX
DESCRIPTOR.message_types_by_name['Item'] = _ITEM
DESCRIPTOR.message_types_by_name['Matrix'] = _MATRIX

Item = _reflection.GeneratedProtocolMessageType('Item', (_message.Message,), dict(
  DESCRIPTOR = _ITEM,
  __module__ = 'source.io.item_pb2'
  # @@protoc_insertion_point(class_scope:music_io.Item)
  ))
_sym_db.RegisterMessage(Item)

Matrix = _reflection.GeneratedProtocolMessageType('Matrix', (_message.Message,), dict(
  DESCRIPTOR = _MATRIX,
  __module__ = 'source.io.item_pb2'
  # @@protoc_insertion_point(class_scope:music_io.Matrix)
  ))
_sym_db.RegisterMessage(Matrix)


# @@protoc_insertion_point(module_scope)