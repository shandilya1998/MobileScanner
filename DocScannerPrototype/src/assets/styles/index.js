import {Platform, StyleSheet, Dimensions, PixelRatio} from "react-native";
const PRIMARY_COLOR = "#7444C0";
const SECONDARY_COLOR = "#5636B8";
const WHITE = "#FFFFFF";
const GRAY = "#757E90";
const DARK_GRAY = "#363636";
const BLACK = "#000000";
const PEACH = "#ffe5b4";
const NAVY_BLUE = "#000080";
const SILVER = "#C0C0C0";

const ONLINE_STATUS = "#46A575";
const OFFLINE_STATUS = "#D04949";

const STAR_ACTIONS = "#FFA200";
const LIKE_ACTIONS = "#B644B2";
const DISLIKE_ACTIONS = "#363636";
const FLASH_ACTIONS = "#5028D7";

const ICON_FONT = "tinderclone";
const DIMENSION_WIDTH = Dimensions.get("window").width;
const DIMENSION_HEIGHT = Dimensions.get("window").height;

export const colors = {
        black: '#1a1917',
        gray: '#888888',
    background1: '#B721FF',
    background2: '#21D4FD'
};

function wp (percentage) {
  const value = (percentage * DIMENSION_WIDTH) / 100; 
  return Math.round(value);
}

const listItemmargin = ((DIMENSION_WIDTH -100)/6 - 30)/2

const itemHorizontalMargin = wp(2);
export const sliderWidth = wp(60);
export const itemWidth = sliderWidth + itemHorizontalMargin * 2; 

export const styles = StyleSheet.create({
    scanner: {
        flex: 1,
        aspectRatio: undefined,
        //justifyContent : 'center',
        //alignSelf : 'center'
    },
    button: {
        alignSelf: 'center',
        //position: 'absolute',
        alignItems: 'center',
        height: 70, 
        justifyContent: 'center',
        width: 65, 
    },
    buttonText: {
        backgroundColor: 'rgba(245, 252, 255, 0.7)',
        fontSize: 32,
    },
    preview: {
        flex: 1,
        width: '100%',
        resizeMode: 'cover',
    },
    permissions: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    cameraButton: {
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'white',
        borderRadius: 50,
        flex: 1,
        margin: 3,
    },
    buttonActionGroup: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'space-between',
    },
    buttonBottomContainer: {
        alignItems: 'flex-end',
        bottom: 40,
        flexDirection: 'row',
        justifyContent: 'space-between',
        left: 25,
        position: 'absolute',
        right: 25,
    },
    buttonContainer: {
        alignItems: 'flex-end',
        //width : DIMENSION_WIDTH,
        bottom: 25,
        flexDirection: 'column',
        justifyContent: 'space-between',
        position: 'absolute',
        right: 25,
        top: 25,
    },
    buttonGroup: {
        backgroundColor: '#00000080',
        borderRadius: 17,
    },
    buttonIcon: {
        color: 'white',
        fontSize: 22,
        marginBottom: 3,
        textAlign: 'center',
    },
    buttonText: {
        color: 'white',
        fontSize: 13,
    },
    buttonTopContainer: {
        alignItems: 'flex-start',
        flexDirection: 'row',
        justifyContent: 'space-between',
        left: 25,
        position: 'absolute',
        right: 25,
        top: 40,
    },
    cameraNotAvailableContainer: {
        alignItems: 'center',
        flex: 1,
        justifyContent: 'center',
        marginHorizontal: 15,
    },
    cameraNotAvailableText: {
        color: 'white',
        fontSize: 25,
        textAlign: 'center',
    },
    cameraOutline: {
        borderColor: 'white',
        borderRadius: 50,
        borderWidth: 3,
        height: 70,
        width: 70,
    },
    container: {
        backgroundColor: 'black',
        flex: 1,
    },
    flashControl: {
        alignItems: 'center',
        borderRadius: 30,
        height: 50,
        justifyContent: 'center',
        margin: 8,
        paddingTop: 7,
        width: 50,
    },
    loadingCameraMessage: {
        color: 'white',
        fontSize: 18,
        marginTop: 10,
        textAlign: 'center',
    },
    loadingContainer: {
        alignItems: 'center', flex: 1, justifyContent: 'center',
    },
    overlay: {
        bottom: 0,
        flex: 1,
        left: 0,
        position: 'absolute',
        right: 0,
        top: 0,
    },
    processingContainer: {
        alignItems: 'center',
        backgroundColor: 'rgba(220, 220, 220, 0.7)',
        borderRadius: 16,
        height: 140,
        justifyContent: 'center',
        width: 200,
    },
    handlerI: {
        borderRadius: 0,
        height: 20,
        width: 20,
    },
    handlerRound: {
        width: 39,
        position: 'absolute',
        height: 39,
        borderRadius: 100,
    },
    image: {
        position: 'absolute',
    },
    bottomButton: {
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'blue',
        width: 70,
        height: 70,
        borderRadius: 100,
    },
    handler: {
        height: 140,
        width: 140,
        overflow: 'visible',
        marginLeft: -70,
        marginTop: -70,
        alignItems: 'center',
        justifyContent: 'center',
        position: 'absolute',
    },
    cropContainer: {
        position: 'absolute',
        left: 0,
        top: 0,
    },
    libraryHeaderTextStyle : {
        color : 'black',
        fontSize : 25
    },
    libraryHeaderTextContainer : {
        flex: 1,
        flexDirection : 'row',
        justifyContent : 'space-between',
        alignItems : 'center',
        padding : 5,
        margin : 5,
    },
    libraryItem : {
        'width' : 110,
        'height' : 120,
    },
    libraryItemIcon : {
        fontSize : 40,

    },
    btn: {
        margin: 2,
        padding: 2,
        backgroundColor: "aqua",
    },
    btnDisable: {
        margin: 2,
        padding: 2,
        backgroundColor: "gray",
    },
    btnText: {
        margin: 2,
        padding: 2,
    }  
});
