import { requireNativeComponent, ViewPropTypes } from 'react-native';
import PropTypes from 'prop-types';

const componentInterface = {
    name: 'RNContrastChangingImageView',
    propTypes: {
    ...ViewPropTypes,
    onSave : PropTypes.func,
    onReset : PropTypes.func,
    contrast : PropTypes.number,
    source : PropTypes.string,
    resizeMode : PropTypes.oneOf(['contain', 'cover', 'stretch'])
    },
};

export default requireNativeComponent('RNContrastChangingImageView', componentInterface);
