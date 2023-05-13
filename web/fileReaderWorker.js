self.onmessage = function(e) {
  const [file, start, size] = e.data;
  const slice = file.slice(start, start + size);

  const reader = new FileReaderSync();
  try {
    const result = reader.readAsArrayBuffer(slice);
    postMessage({ status: 'success', result });
  } catch (e) {
    postMessage({ status: 'error', error: e.toString() });
  }
};
