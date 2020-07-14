module.exports = {
  assets: ['./Fonts/'],
  dependencies: {
    'react-native-perspective-image-cropper': {
      platforms: {
        android: {
          packageImportPath: 'import fr.michaelvilleneuve.customcrop.RNCustomCropPackage;',
        }
      }
    }
  }
};
