import React from 'react';
import ConfigGenerator from '../../base/ConfigGenerator';

/**
 * Llama 3.2 Configuration Generator
 */
const Llama32ConfigGenerator = () => {
  const config = {
    modelFamily: 'llama-3.2',

    options: {
      hardware: {
        name: 'hardware',
        title: 'Hardware Platform',
        items: [
          { id: 'cpu', label: 'Xeon CPU', default: true }
        ]
      },
      quantization: {
        name: 'quantization',
        title: 'Quantization',
        items: [
          { id: 'w8a8', label: 'W8A8', default: true },
          { id: 'fp8', label: 'FP8', default: false },
          { id: 'awq', label: 'AWQ', default: false },
        ]
      },
    },

    generateCommand: function(values) {
      const { hardware, quantization } = values;

      // Determine model path based on current model_id_list
      let modelPath;
      if (quantization === 'w8a8') {
        modelPath = 'RedHatAI/Llama-3.2-3B-quantized.w8a8';
      } else if (quantization === 'fp8') {
        modelPath = 'RedHatAI/Llama-3.2-3B-Instruct-FP8';
      } else if (quantization === 'awq') {
        modelPath = 'AMead10/Llama-3.2-3B-Instruct-AWQ';
      } else {
        // Default fallback if needed, though current list covers the provided options
        modelPath = 'RedHatAI/Llama-3.2-3B-quantized.w8a8';
      }

      // Build command args
      const args = [];
      args.push(`--model-path ${modelPath}`);
      args.push(`--trust-remote-code`);
      args.push(`--disable-overlap-schedule`);
      args.push(`--device cpu`);
      
      if (quantization === 'w8a8') {
        args.push(`--quantization w8a8_int8`);
      }
      // Note: FP8 and AWQ support in SGLang CPU depends on specific backend implementations
      
      args.push(`--enable-torch-compile`);
      args.push(`--host 0.0.0.0`);
      args.push(`--tp 1`); // Adjusted TP for 3B model as a sane default

      let cmd = 'python -m sglang.launch_server \\\n';
      cmd += `  ${args.join(' \\\n  ')}`;

      return cmd;
    }
  };

  return <ConfigGenerator config={config} />;
};

export default Llama32ConfigGenerator;