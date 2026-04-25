import React from 'react';
import ConfigGenerator from '../../base/ConfigGenerator';

/**
 * Llama 3.2 Configuration Generator
 */
const Llama32ConfigGenerator = () => {
  const config = {
    modelFamily: 'meta-llama',

    options: {
      hardware: {
        name: 'hardware',
        title: 'Hardware Platform',
        items: [
          { id: 'cpu', label: 'Xeon CPU', default: true }
        ]
      },
      modelsize: {
        name: 'modelsize',
        title: 'Model Size',
        items: [
          { id: '3b', label: '3B', default: true }
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
      const { hardware, modelsize, quantization } = values;

      // Determine model path based on current model_list
      let modelPath;
      if (quantization === 'w8a8') {
        modelPath = `Llama-3.2-3B-quantized.w8a8`;
      } else if (quantization === 'fp8') {
        modelPath = `Llama-3.2-3B-Instruct-FP8`;
      } else if (quantization === 'awq') {
        modelPath = `Llama-3.2-3B-Instruct-AWQ`;
      } else {
        modelPath = `Llama-3.2-3B`;
      }

      // Build command args
      const args = [];
      args.push(`--model-path ${modelPath}`);
      args.push(`--trust-remote-code`);
      args.push(`--disable-overlap-schedule`);
      args.push(`--device cpu`);
      
      if (quantization === 'w8a8') {
        args.push(`--quantization w8a8`);
      }
      
      args.push(`--enable-torch-compile`);
      args.push(`--host 0.0.0.0`);
      args.push(`--tp 1`); // Adjust TP as needed for 3B models

      let cmd = 'python -m sglang.launch_server \\\n';
      cmd += `  ${args.join(' \\\n  ')}`;

      return cmd;
    }
  };

  return <ConfigGenerator config={config} />;
};

export default Llama32ConfigGenerator;