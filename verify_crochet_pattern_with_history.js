#!/usr/bin/env node

/**
 * Crochet Pattern Validator With History Support
 * 
 * Enhanced version that can track history of step validation
 * It takes a pattern string as input and outputs whether it's valid or not.
 * USAGE:
 * node verify_crochet_pattern_with_history.js [pattern_string_or_file]
 * 
 * Set INCLUDE_HISTORY=1 environment variable to include history information
 */

// Import required modules
const fs = require('fs');
const path = require('path');

// Get the path to parse57.js
const parserPath = path.join(__dirname, 'CrochetPARADE', 'parse57.js');
// Check if parse57.js exists
if (!fs.existsSync(parserPath)) {
  console.error(`Error: Could not find the CrochetPARADE parser at ${parserPath}`);
  process.exit(1);
}

/**
 * Create a mock window object with required properties
 * to simulate the browser environment needed by parse57.js
 */
global.window = {
  alert: (msg) => { console.error(`Pattern Error: ${msg}`); }
};

// Mock console.trace to avoid stack trace output
const originalConsoleTrace = console.trace;
console.trace = () => {};

// Mock document object
global.document = {
  getElementById: () => ({
    value: '',
    style: {}
  }),
  createElement: () => ({
    style: {},
    appendChild: () => {}
  }),
  createTextNode: () => ({}),
  querySelector: () => ({ innerHTML: '' }),
  querySelectorAll: () => []
};

// Load the parser
const parserCode = fs.readFileSync(parserPath, 'utf8');

// Create a function context for the parser code
const funcContext = {
  window,
  document,
  console,
  setTimeout: setTimeout,
  clearTimeout: clearTimeout,
  localStorage: {
    getItem: () => null,
    setItem: () => {}
  },
  fetch: () => Promise.resolve({
    json: () => Promise.resolve({})
  })
};

// Execute the parser code in the global context to make functions available
eval(parserCode);

// Check if history tracking is enabled via environment variable
const includeHistory = process.env.INCLUDE_HISTORY === '1';

/**
 * Validate a crochet pattern with history support
 * 
 * @param {string} patternString - The crochet pattern to validate
 * @return {object} - Validation result with optional history
 */
function validatePatternWithHistory(patternString) {
  // Reset errors
  let errors = [];
  let warnings = [];
  
  // Override alert to capture errors
  window.alert = (msg) => {
    errors.push(msg);
  };
  
  // Override console.warn to capture warnings
  const originalConsoleWarn = console.warn;
  console.warn = (msg) => {
    warnings.push(msg);
  };
  
  try {
    // Check if the necessary functions are available from parse57.js
    if (typeof final !== 'function') {
      throw new Error('Could not access the required functions from parse57.js');
    }
    
    // Split the pattern into lines for history analysis
    const lines = patternString.split('\n');
    let history = [];
    
    // Process each line progressively if history is requested
    if (includeHistory) {
      for (let i = 0; i < lines.length; i++) {
        // Build pattern up to this line
        const partialPattern = lines.slice(0, i + 1).join('\n');
        
        // Reset errors for this iteration
        let stepErrors = [];
        const stepAlert = (msg) => {
          stepErrors.push(msg);
        };
        window.alert = stepAlert;
        
        // Try to parse this partial pattern
        try {
          const stepResult = final(partialPattern);
          history.push({
            line_number: i + 1,
            line_content: lines[i],
            is_valid: stepErrors.length === 0,
            errors: stepErrors.length > 0 ? stepErrors : undefined,
            step_result: stepResult ? {
              stitchCount: stepResult.Stitches ? stepResult.Stitches.length : 0,
              rowCount: stepResult.Stitches ? Math.max(...stepResult.Stitches.map(s => s.nrow).filter(n => !isNaN(n)), 0) + 1 : 0
            } : undefined
          });
        } catch (stepError) {
          history.push({
            line_number: i + 1,
            line_content: lines[i],
            is_valid: false,
            errors: stepErrors.length > 0 ? stepErrors : [`Exception: ${stepError.message}`],
            exception: stepError.message
          });
        }
      }
    }
    
    // Reset errors for the full pattern validation
    errors = [];
    window.alert = (msg) => {
      errors.push(msg);
    };
    
    // Try to parse the pattern using the main parsing function from parse57.js
    const result = final(patternString);
    
    // Restore console functions
    console.warn = originalConsoleWarn;
    console.trace = originalConsoleTrace;
    
    // Build the response object
    const response = {
      valid: errors.length === 0,
      message: errors.length === 0 ? 'Pattern is valid and can be compiled by CrochetPARADE' : 'Pattern has errors',
      warnings: warnings.length > 0 ? warnings : undefined,
      details: result ? {
        stitchCount: result.Stitches ? result.Stitches.length : 0,
        rowCount: result.Stitches ? Math.max(...result.Stitches.map(s => s.nrow).filter(n => !isNaN(n)), 0) + 1 : 0
      } : undefined
    };
    
    // Add errors if any
    if (errors.length > 0) {
      response.errors = errors;
    }
    
    // Add history if requested
    if (includeHistory && history.length > 0) {
      response.history = history;
    }
    
    return response;
  } catch (error) {
    // Restore console functions
    console.warn = originalConsoleWarn;
    console.trace = originalConsoleTrace;
    
    return {
      valid: false,
      errors: errors.length > 0 ? errors : [`Exception during pattern parsing: ${error.message}`],
      exception: error.message,
      history: includeHistory ? [{
        line_number: 1,
        is_valid: false,
        errors: [`Exception during pattern parsing: ${error.message}`],
        exception: error.message
      }] : undefined
    };
  }
}

/**
 * Main function
 */
function main() {
  // Check if a pattern file was provided
  if (process.argv.length < 3) {
    // Check if we're receiving input from a pipe
    if (!process.stdin.isTTY) {
      let patternString = '';
      
      // Read from stdin
      process.stdin.on('data', (chunk) => {
        patternString += chunk;
      });
      
      process.stdin.on('end', () => {
        const result = validatePatternWithHistory(patternString);
        console.log(JSON.stringify(result, null, 2));
        process.exit(result.valid ? 0 : 1);
      });
      
      return;
    }
    
    // No input provided
    console.log('Usage: node verify_crochet_pattern_with_history.js [pattern_string_or_file]');
    console.log('   or: cat pattern.txt | node verify_crochet_pattern_with_history.js');
    process.exit(1);
  }
  
  const arg = process.argv[2];
  
  // Check if the argument is a file path
  if (fs.existsSync(arg) && fs.statSync(arg).isFile()) {
    try {
      const patternString = fs.readFileSync(arg, 'utf8');
      const result = validatePatternWithHistory(patternString);
      console.log(JSON.stringify(result, null, 2));
      process.exit(result.valid ? 0 : 1);
    } catch (error) {
      console.error(`Error reading file: ${error.message}`);
      process.exit(1);
    }
  } else {
    // Treat the argument as the pattern string
    const result = validatePatternWithHistory(arg);
    console.log(JSON.stringify(result, null, 2));
    process.exit(result.valid ? 0 : 1);
  }
}

// Execute the main function
main();
