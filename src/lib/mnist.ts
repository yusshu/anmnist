/*! decoder for the mnist dataset */
const IMAGES_DATA_MAGIC_NUMBER = 2051;
const LABELS_DATA_MAGIC_NUMBER = 2049;

export interface ReadableObject {
  open(): Promise<Buffer>;
}

export interface MnistEntry {
  label: number;
  image: number[];
}

/**
 * Decode the given images and labels input streams and return an array of arrays of images.
 *
 * @param imagesInput the input stream for the images file
 * @param labelsInput the input stream for the labels file
 * @param limit the maximum number of samples to read
 */
export async function decodeMnist(imagesInput: ReadableObject, labelsInput: ReadableObject, limit = Infinity): Promise<MnistEntry[]> {
  // read labels first
  const labelsData = await labelsInput.open();

  // validate labels magic number
  {
    const magicNumber = labelsData.readUInt32BE(0);
    if (magicNumber !== LABELS_DATA_MAGIC_NUMBER) {
      console.error(`Invalid magic number: ${magicNumber} got from labels data (${labelsData}),`
          + ` expected ${LABELS_DATA_MAGIC_NUMBER}`);
      return [];
    }
  }

  // read size in labels
  const labelsSize = labelsData.readUInt32BE(4);

  // read labels
  const labels = new Uint8Array(labelsSize);
  for (let i = 0; i < Math.min(limit, labelsSize); i++) {
    labels[i] = labelsData.readUInt8(8 + i);
  }

  // now read the images data
  const imagesData = await imagesInput.open();

  // validate magic number
  {
    const magicNumber = imagesData.readUInt32BE(0);
    if (magicNumber !== IMAGES_DATA_MAGIC_NUMBER) {
      console.error(`Invalid magic number: ${magicNumber} got from images data (${imagesData}),`
          + ` expected ${IMAGES_DATA_MAGIC_NUMBER}`);
      return [];
    }
  }

  const imagesSize = imagesData.readUInt32BE(4);
  if (imagesSize !== labelsSize) {
    console.error(`Images size (${imagesSize}) is different from labels size (${labelsSize})`);
    return [];
  }

  const rows = imagesData.readUInt32BE(8);
  const cols = imagesData.readUInt32BE(12);
  const digitBits = rows * cols;
  const output: MnistEntry[] = [];

  for (let i = 0; i < Math.min(imagesSize, limit); i++) {
    const image = new Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        image[r * cols + c] = imagesData.readUInt8(16 + i * digitBits + r * cols + c) / 255;
      }
    }

    const label = labels[i];
    output.push({ label, image });
  }

  return output;
}

export function decodeMnistFromUrls(imagesUrl: string, labelsUrl: string, limit = Infinity): Promise<MnistEntry[]> {
  // use fetch to create buffer from the file located at the given urls
  function readableObjectFromURL(url: string): ReadableObject {
    return {
      open: async () => {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        return Buffer.from(buffer);
      }
    };
  }

  return decodeMnist(readableObjectFromURL(imagesUrl), readableObjectFromURL(labelsUrl), limit);
}