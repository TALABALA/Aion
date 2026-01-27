# Example Tool Plugin

A comprehensive example demonstrating AION plugin development.

## Overview

This plugin showcases how to build a tool plugin for AION, including:
- Tool definitions with typed parameters
- Configuration handling with schema validation
- Hook handlers for extending AION behavior
- Proper lifecycle management
- Health checks

## Installation

Copy this directory to your AION plugins folder:
```bash
cp -r example-tool ~/.aion/plugins/
```

Or symlink for development:
```bash
ln -s $(pwd)/example-tool ~/.aion/plugins/
```

## Tools Provided

### `greet`
Generate a personalized greeting message.

**Parameters:**
- `name` (string, required): Name of the person to greet
- `formal` (boolean, optional): Use formal greeting style

**Example:**
```json
{
  "tool": "greet",
  "params": {
    "name": "Alice",
    "formal": false
  }
}
```

### `calculate`
Perform basic arithmetic calculations.

**Parameters:**
- `operation` (string, required): One of "add", "subtract", "multiply", "divide"
- `a` (number, required): First operand
- `b` (number, required): Second operand

**Example:**
```json
{
  "tool": "calculate",
  "params": {
    "operation": "multiply",
    "a": 6,
    "b": 7
  }
}
```

### `hash_text`
Generate a hash of the input text.

**Parameters:**
- `text` (string, required): Text to hash
- `algorithm` (string, optional): Hash algorithm (md5, sha1, sha256, sha512)

### `reverse_text`
Reverse a string.

**Parameters:**
- `text` (string, required): Text to reverse
- `by_word` (boolean, optional): Reverse word order instead of characters

## Configuration

```json
{
  "greeting_prefix": "Hello",
  "enable_logging": false,
  "max_retries": 3
}
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `greeting_prefix` | string | "Hello" | Prefix for greeting messages |
| `enable_logging` | boolean | false | Enable verbose logging |
| `max_retries` | integer | 3 | Maximum retry attempts |

## Hooks

This plugin registers handlers for:

- `tool.before_execute`: Logs tool executions when logging is enabled
- `request.before`: Adds metadata to incoming requests

## Development

To modify this plugin:

1. Edit `plugin.py` to add new tools or modify existing ones
2. Update `manifest.json` with any new configuration or permissions
3. Reload the plugin:

```python
await plugin_manager.reload("example-tool")
```

## License

MIT
