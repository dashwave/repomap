(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_signature
  name: (identifier) @name.definition.function_signature) @definition.function_signature

(import_or_export
  (library_import (
    import_specification (
      configurable_uri (
        uri
      ) @name.import
    )
  ))
)
