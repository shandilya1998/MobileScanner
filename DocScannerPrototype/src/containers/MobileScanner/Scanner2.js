import { PropTypes } from 'prop-types';
import React, { PureComponent } from 'react';
import { ActivityIndicator, Switch, Animated, Dimensions, Platform, SafeAreaView, StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import Scanner, { Filters, RectangleOverlay } from 'react-native-rectangle-scanner';
//import ScannerFilters from './Filters';
import {styles} from '../../assets/styles';
import {connect} from 'react-redux';
import {onPictureProcessed} from '../../actions/actions';

class MobileScanner extends PureComponent {
    static propTypes = {
        cameraIsOn: PropTypes.bool,
        onLayout: PropTypes.func,
        onSkip: PropTypes.func,
        onCancel: PropTypes.func,
        onPictureTaken: PropTypes.func,
        hideSkip: PropTypes.bool,
        initialFilterId: PropTypes.number,
    }

    static defaultProps = {
        cameraIsOn: undefined,
        onLayout: () => {},
        onSkip: () => {},
        onCancel: () => {},
        onPictureTaken: () => {},
        hideSkip: false,
    }

    constructor(props) {
        super(props);
        this.state = {
            flashEnabled: false,
            showScannerView: false,
            didLoadInitialLayout: false,
            detectedRectangle: false,
            candidateRectangle : undefined,
            isMultiTasking: false,
            loadingCamera: true,
            captureMultiple : false,
            processingImage: false,
            takingPicture: false,
            overlayFlashOpacity: new Animated.Value(0),
            device: {
                initialized: false,
                hasCamera: false,
                permissionToUseCamera: false,
                flashIsAvailable: false,
                previewHeightPercent: 1,
                previewWidthPercent: 1,
            },
        };

        this.camera = React.createRef();
        this.imageProcessorTimeout = null;
        this.onDeviceSetup = this.onDeviceSetup.bind(this);
        this.getCameraDisabledMessage = this.getCameraDisabledMessage.bind(this);
        this.toggleSwitch = this.toggleSwitch.bind(this);    
        this.onPictureProcessed = this.onPictureProcessed.bind(this);
        this.onPictureTaken = this.onPictureTaken.bind(this);
        this.turnOnCamera = this.turnOnCamera.bind(this);
        this.onPressDone = this.onPressDone.bind(this);
    }

    componentDidMount() {
        if (this.state.didLoadInitialLayout && !this.state.isMultiTasking) {
            this.turnOnCamera();
        }
    }

    componentDidUpdate() {
        if (this.state.didLoadInitialLayout) {
            if (this.state.isMultiTasking) return this.turnOffCamera(true);
            if (this.state.device.initialized) {
                if (!this.state.device.hasCamera) return this.turnOffCamera();
                if (!this.state.device.permissionToUseCamera) return this.turnOffCamera();
            }

            if (!this.state.showScannerView) {
                return this.turnOnCamera();
            }
        }
        return null;
    }

    componentWillUnmount() {
        clearTimeout(this.imageProcessorTimeout);
    }

    // Called after the device gets setup. This lets you know some platform specifics
    // like if the device has a camera or flash, or even if you have permission to use the
    // camera. It also includes the aspect ratio correction of the preview
    onDeviceSetup(deviceDetails){
        const {
            hasCamera, permissionToUseCamera, flashIsAvailable, previewHeightPercent, previewWidthPercent,
        } = deviceDetails;
        this.setState({
            loadingCamera: false,
            device: {
                initialized: true,
                hasCamera,
                permissionToUseCamera,
                flashIsAvailable,
                previewHeightPercent: previewHeightPercent || 1,
                previewWidthPercent: previewWidthPercent || 1,
            },
        });
    }

    // Determine why the camera is disabled.
    getCameraDisabledMessage() {
        if (this.state.isMultiTasking) {
            return 'Camera is not allowed in multi tasking mode.';
        }

        const { device } = this.state;
        if (device.initialized) {
            if (!device.hasCamera) {
                return 'Could not find a camera on the device.';
            }
            if (!device.permissionToUseCamera) {
                return 'Permission to use camera has not been granted.';
            }
        }
        return 'Failed to set up the camera.';
    }

    // On some android devices, the aspect ratio of the preview is different than
    // the screen size. This leads to distorted camera previews. This allows for correcting that.
    getPreviewSize() {
        const dimensions = Dimensions.get('window');
        // We use set margin amounts because for some reasons the percentage values don't align the camera preview in the center correctly.
        const heightMargin = (1 - this.state.device.previewHeightPercent) * dimensions.height / 2;
        const widthMargin = (1 - this.state.device.previewWidthPercent) * dimensions.width / 2;
        if (dimensions.height > dimensions.width) {
            // Portrait
            return {
                height: this.state.device.previewHeightPercent,
                width: this.state.device.previewWidthPercent,
                marginTop: heightMargin,
                marginLeft: widthMargin,
            };
        }

        // Landscape
        return {
            width: this.state.device.previewHeightPercent,
            height: this.state.device.previewWidthPercent,
            marginTop: widthMargin,
            marginLeft: heightMargin,
        };
    }

    // Capture the current frame/rectangle. Triggers the flash animation and shows a
    // loading/processing state. Will not take another picture if already taking a picture.
    capture = () => {
        if (this.state.takingPicture) return;
        if (this.state.processingImage) return;
        const {detectedRectangle} = this.state;
        this.setState({ 
            takingPicture: true, 
            processingImage: true, 
            candidateRectangle : detectedRectangle,
        });
        this.camera.current.capture();
        this.triggerSnapAnimation();
        //this.turnOffCamera();
        // If capture failed, allow for additional captures
        this.imageProcessorTimeout = setTimeout(() => {
            if (this.state.takingPicture) {
                this.setState({ takingPicture: false });
            }
        }, 100);
    }

    // The picture was captured but still needs to be processed.
    onPictureTaken(event){
        this.setState({ takingPicture: false });
        this.props.onPictureTaken(event);
        //this.camera.current.stop();
    }

    // The picture was taken and cached. You can now go on to using it.
    onPictureProcessed(event){
        //console.log(event);
        this.props.onPictureProcessed({originalImage : event.initialImage,
                                       detectedImage : event.croppedImage, 
                                       rectCoords : this.state.candidateRectangle});
        this.setState({
            takingPicture: false,
            processingImage: false,
            showScannerView: this.state.captureMultiple ? true : false,
            //cameraLoading : false,
        });
        if(!this.state.captureMultiple){
            this.props.navigation.navigate('edit',
                                           {'captureMultiple' : this.state.captureMultiple});
        }
        else{
            this.camera.current.refresh();
        }
    }

    // Flashes the screen on capture
    triggerSnapAnimation() {
        Animated.sequence([
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0.2, duration: 100 }),
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0, duration: 50 }),
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0.6, delay: 100, duration: 120 }),
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0, duration: 90 }),]).start();
    }

    // Hides the camera view. If the camera view was shown and onDeviceSetup was called,
    // but no camera was found, it will not uninitialize the camera state.
    turnOffCamera(shouldUninitializeCamera = false) {
        if (shouldUninitializeCamera && this.state.device.initialized) {
            this.setState(({ device }) => ({
                showScannerView: false,
                device: { ...device, initialized: false },
            }));
        } else if (this.state.showScannerView) {
            this.setState({ showScannerView: false, loadingCamera : false });
        }
        //this.camera.current.stop();
    } 

    // Will show the camera view which will setup the camera and start it.
    // Expect the onDeviceSetup callback to be called
    turnOnCamera() {
        if (!this.state.showScannerView) {
            //this.camera.start();        
            this.setState({
                showScannerView: true,
                loadingCamera: true,
            });
        }
    }

    // Renders the flashlight button. Only shown if the device has a flashlight.
    renderFlashControl() {
        const { flashEnabled, device } = this.state;
        if (!device.flashIsAvailable) return null;
        return (
            <TouchableOpacity
                style={[
                    styles.flashControl, 
                    { backgroundColor: flashEnabled ? '#FFFFFF80' : '#00000080' }]}
                activeOpacity={0.8}
                onPress={() => this.setState({ flashEnabled: !flashEnabled })}>
                <Icon 
                    name="ios-flashlight" 
                    style={[
                        styles.buttonIcon, 
                        { fontSize: 28, color: flashEnabled ? '#333' : '#FFF' }]} />
            </TouchableOpacity>
        );
    }

    toggleSwitch(){
        const value = this.state.captureMultiple;
        this.setState({captureMultiple : !value});
    }

    onPressDone(){
        console.log(this.prop);
        this.props.navigation.navigate(
            'edit',
            {'captureMultiple' : this.state.captureMultiple}
        );    
    }

    renderDoneButton(){
        if (!this.state.captureMultiple) return null;
        return (
            <TouchableOpacity
                style={[
                    styles.flashControl, 
                    { backgroundColor: '#00000080' }]} 
                activeOpacity={0.8}
                onPress={() => {this.onPressDone()}}>
                <Icon 
                    name="md-done-all" 
                    style={[
                        styles.buttonIcon, 
                        { fontSize: 28, color: '#FFF' }]} />
            </TouchableOpacity>
        ); 
    }

    // Renders the camera controls. This will show controls on the side for large tablet screens
    // or on the bottom for phones. (For small tablets it will adjust the view a little bit).
    renderCameraControls(){
        const dimensions = Dimensions.get('window');
        const aspectRatio = dimensions.height / dimensions.width;
        const isPhone = aspectRatio > 1.6;
        const cameraIsDisabled = this.state.takingPicture || this.state.processingImage;
        const disabledStyle = { opacity: cameraIsDisabled ? 0.8 : 1 };
        if (!isPhone) {
            if (dimensions.height < 500) {
                return (
                    <View style={styles.buttonContainer}>
                        <View 
                            style={[
                                styles.buttonActionGroup, 
                                { 
                                    flexDirection: 'row', 
                                    alignItems: 'flex-end', 
                                    marginBottom: 28 }]}>
                            {this.renderDoneButton()}
                            {this.renderFlashControl()}
                            <View style={[
                                styles.buttonGroup, 
                                { marginLeft: 8 }]}>
                                <View style= {styles.button}>
                                    <Switch
                                        trackColor={{ false: "#767577", true: "#81b0ff" }}
                                        thumbColor={this.state.captureMultiple ? "#f5dd4b" : "#f4f3f4"}
                                        ios_backgroundColor="#3e3e3e"
                                        onValueChange={this.toggleSwitch}
                                        value={this.state.captureMultiple}
                                        style={{
                                            paddingHorizontal: 14, 
                                            paddingVertical: 13, 
                                            height: 50, 
                                            width: 50, 
                                        }}/>
                                    <Text style={styles.buttonText}>Multiple</Text>
                                </View>
                            </View>
                        </View>
                        <View style={[styles.cameraOutline, disabledStyle]}>
                            <TouchableOpacity
                                activeOpacity={0.8}
                                style={styles.cameraButton}
                                onPress={this.capture}/>
                        </View>
                        <View 
                            style={[
                                styles.buttonActionGroup, 
                                { marginTop: 28 }]}>
                            <View style={styles.buttonGroup}>
                                <TouchableOpacity
                                    style={styles.button}
                                    onPress={this.props.onCancel}
                                    activeOpacity={0.8}>
                                    <Icon 
                                        name="md-photos" 
                                        size={40} 
                                        style={styles.buttonIcon} />
                                    <Text style={styles.buttonText}>Photos</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    </View>
                );
            }
            return (
                <View style={styles.buttonContainer}>
                    <View 
                        style={[
                            styles.buttonActionGroup, 
                            { 
                                justifyContent: 'flex-end', 
                                marginBottom: 20 }]}>
                        {this.renderFlashControl()}
                        <View style={{ 
                            flexDirection: 'column', 
                            justifyContent: 'flex-end' }}> 
                            {null}
                        </View>
                    </View>
                    <View style={[styles.cameraOutline, disabledStyle]}>
                        <TouchableOpacity
                            activeOpacity={0.8}
                            style={styles.cameraButton}
                            onPress={this.capture}/>
                    </View>
                    <View 
                        style={[
                            styles.buttonActionGroup, 
                            { marginTop: 28 }]}>
                        {this.renderDoneButton()}
                        <View style={styles.buttonGroup}>
                            <View style = {styles.button}>
                                <Switch
                                    trackColor={{ false: "#767577", true: "#81b0ff" }}
                                    thumbColor={this.state.captureMultiple ? "#f5dd4b" : "#f4f3f4"}
                                    ios_backgroundColor="#3e3e3e"
                                    onValueChange={this.toggleSwitch}
                                    value={this.state.captureMultiple}
                                    style={{
                                        paddingHorizontal: 14, 
                                        paddingVertical: 13, 
                                        height: 50, 
                                        width: 50, 
                                    }}/>
                                <Text style={styles.buttonText}>Multiple</Text>
                        
                            </View>
                        </View>
                        <View style={styles.buttonGroup}>
                            <TouchableOpacity
                                style={styles.button}
                                onPress={this.props.onCancel}
                                activeOpacity={0.8}>
                                <Icon 
                                    name="md-photos" 
                                    size={40} 
                                    style={styles.buttonIcon} />
                                <Text style={styles.buttonText}>Photos</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            );
        }

        return (
            <>
                <View style={styles.buttonBottomContainer}>
                    <View style={styles.buttonGroup}>
                        <TouchableOpacity
                            style={styles.button}
                            onPress={this.props.onCancel}
                            activeOpacity={0.8}>
                            <Icon 
                                name="md-photos" 
                                size={40} 
                                style={styles.buttonIcon} />
                            <Text style={styles.buttonText}>Photos</Text>
                        </TouchableOpacity>
                    </View>
                    <View style={[styles.cameraOutline, disabledStyle]}>
                        <TouchableOpacity
                            activeOpacity={0.8}
                            style={styles.cameraButton}
                            onPress={this.capture}/>
                    </View>
                    <View>
                        <View 
                            style={[
                                styles.buttonActionGroup, 
                                { 
                                    justifyContent: 'flex-end', 
                                    marginBottom : 16 
                                }]}>
                            <View style={{ 
                                flexDirection: 'column', 
                                justifyContent: 'flex-end' }}> 
                                {null}
                            </View>
                            {this.renderFlashControl()}
                        </View>
                        <View style={styles.buttonGroup}>
                            {this.renderDoneButton()}
                            <View
                                style = {styles.button}>
                                <Switch
                                    trackColor={{ false: "#767577", true: "#81b0ff" }}
                                    thumbColor={this.state.captureMultiple ? "#f5dd4b" : "#f4f3f4"}
                                    ios_backgroundColor="#3e3e3e"
                                    onValueChange={this.toggleSwitch}
                                    value={this.state.captureMultiple}
                                    style={{
                                        marginBottom : 3,
                                        alignSelf : 'center',
                                    }}/>     
                                <Text style={styles.buttonText}>Multiple</Text>
                            </View>
                        </View>
                    </View>
                </View>
            </>
        );
    }

    // Renders the camera controls or a loading/processing state
    renderCameraOverlay(){
        let loadingState = null;
        if (this.state.loadingCamera) {
            //console.log('this');
            loadingState = (
                <View 
                    style={[
                        styles.overlay, 
                        {
                            backgroundColor : 'black',
                            //width : Dimensions.get('window').width, 
                        }]}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator color="white" />
                        <Text style={styles.loadingCameraMessage}>Loading Camera</Text>
                    </View>
                </View>
            );
        } else if (this.state.processingImage) {
            loadingState = (
                <View style={[styles.overlay, {backgroundColor : 'black'}]}>
                    <View style={styles.loadingContainer}>
                        <View style={styles.processingContainer}>
                            <ActivityIndicator color="#333333" size="large" />
                            <Text style={{ color: '#333333', fontSize: 30, marginTop: 10 }}>Processing</Text>
                        </View>
                    </View>
                </View>
            );
        }

        return (
            <SafeAreaView
                style = {[
                    styles.overlay,
                    {
                        //width : Dimensions.get('window').width,
                        //alignSelf : 'center',
                        alignItems : 'center',
                        //justifyContent : 'center',
                        //flexDirection : 'column',
                    }
                ]}>
                {loadingState}
                <View 
                    style={[
                        styles.overlay,
                        {
                            position : 'relative',
                            //flex : 1,
                            width : Dimensions.get('window').width,
                            //alignItems : 'center',
                            //alignSelf : 'center',
                            //backgroundColor : 'red',
                        }  
                    ]}>
                    {this.renderCameraControls()}
                </View>
            </SafeAreaView>
        );
    }

    // Renders either the camera view, a loading state, or an error message
    // letting the user know why camera use is not allowed
    renderCameraView() {
        if (this.state.showScannerView) {
            let previewSize = this.getPreviewSize();
            previewSize.width = 1/previewSize.height;
            previewSize.height = 1;
            let rectangleOverlay = null;
            if (!this.state.loadingCamera && !this.state.processingImage) {
                rectangleOverlay = (
                    <RectangleOverlay
                        detectedRectangle={this.state.detectedRectangle}
                        previewRatio={previewSize}
                        backgroundColor="rgba(255,181,6, 0.2)"
                        borderColor="rgb(255,181,6)"
                        borderWidth={4}
                        // == These let you auto capture and change the overlay style on detection ==
                        // detectedBackgroundColor="rgba(255,181,6, 0.3)"
                        // detectedBorderWidth={6}
                        // detectedBorderColor="rgb(255,218,124)"
                        // onDetectedCapture={this.capture}
                        // allowDetection
                    />
                );
            } 

            // NOTE: I set the background color on here because for some reason the view doesn't line up correctly otherwise. It's a weird quirk I noticed.
            return (
                <View 
                    style={{ 
                        //flex : 1,
                        backgroundColor: 'rgba(0, 0, 0, 0)', 
                        position: 'relative', 
                        //justifyContent : 'center',
                        alignSelf : 'center',
                        margin:  previewSize.marginTop, 
                        margin: previewSize.marginLeft,
                        height: `${previewSize.height * 100}%`, 
                        width: `${previewSize.width * 100}%`,
                    }}> 
                    <Scanner
                        onErrorProcessingImage = {(err)=>console.log(err)}
                        onPictureTaken={this.onPictureTaken}
                        onPictureProcessed={this.onPictureProcessed}
                        enableTorch={this.state.flashEnabled}
                        ref={this.camera}
                        capturedQuality={0.6}
                        onRectangleDetected={({ detectedRectangle }) => {this.setState({ detectedRectangle })}}
                        onDeviceSetup={this.onDeviceSetup}
                        onTorchChanged={({ enabled }) => this.setState({ flashEnabled: enabled })}
                        style={styles.scanner}/>
                    {rectangleOverlay}
                    <Animated.View 
                        useNativeDriver = {true}
                        style={{ 
                            ...styles.overlay, 
                            backgroundColor: 'white', 
                            opacity: this.state.overlayFlashOpacity }} />
                    {this.renderCameraOverlay()}
                </View>
            );
        }

        let message = null;
        if (this.state.loadingCamera) {
            //console.log('this 2');
            message = (
                <View style={styles.overlay}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator color="white" />
                        <Text style={styles.loadingCameraMessage}>Loading Camera</Text>
                    </View>
                </View>
            );
        } else {
            message = (
                <Text style={styles.cameraNotAvailableText}>
                    {this.getCameraDisabledMessage()}
                </Text>
            );
        }

        return (
            <View style={styles.cameraNotAvailableContainer}>
                {message}
                <View style={styles.buttonBottomContainer}>
                    <View style={styles.buttonGroup}>
                        <TouchableOpacity
                            style={styles.button}
                            onPress={()=>{}}
                            activeOpacity={0.8}>
                            <Icon 
                                name="ios-close-circle" 
                                size={40} 
                                style={styles.buttonIcon} />
                            <Text style={styles.buttonText}>Cancel</Text>
                        </TouchableOpacity>
                    </View>
                    <View style={styles.buttonGroup}>
                        <TouchableOpacity
                            style={[styles.button, { marginTop: 8 }]}
                            onPress={this.props.onSkip}
                            activeOpacity={0.8}>
                            <Icon 
                                name="md-arrow-round-forward" 
                                size={40} 
                                color="white" 
                                style={styles.buttonIcon} />
                            <Text style={styles.buttonText}>Skip</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        );
    }

    render() {
        return (
            <View
                style={styles.container}
                onLayout={(event) => {
                // This is used to detect multi tasking mode on iOS/iPad
                // Camera use is not allowed
                this.props.onLayout(event);
                if (this.state.didLoadInitialLayout && Platform.OS === 'ios') {
                    const screenWidth = Dimensions.get('screen').width;
                    const isMultiTasking = (
                    Math.round(event.nativeEvent.layout.width) < Math.round(screenWidth)
                );
                    if (isMultiTasking) {
                        this.setState({ isMultiTasking: true, loadingCamera: false });
                    } else {
                        this.setState({ isMultiTasking: false });
                    }
                } else {
                    this.setState({ didLoadInitialLayout: true });
                }}}>
                <StatusBar backgroundColor="black" barStyle="light-content" hidden={true} />
                {this.renderCameraView()}
            </View>
        );
    }
}

const mapStateToProps = (state) => {
    return {};
};

const mapDispatchToProps = (dispatch) => {
    return {
        onPictureProcessed : (image) => {dispatch(onPictureProcessed(image))},
    };
};
export default connect(mapStateToProps, mapDispatchToProps)(MobileScanner);
