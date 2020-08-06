import React, {Component} from 'react';
import {Platform} from 'react-native';
import {createDrawerNavigator } from '@react-navigation/drawer';
import {NavigationContainer} from '@react-navigation/native';
import {navigatorRef} from './rootNavigator';
import MobileScanner from './containers/MobileScanner';
import ScanContainer from './containers/ScanContainer';
import {checkMultiple, 
        PERMISSIONS,
        request} from 'react-native-permissions';

MainStack = createDrawerNavigator();
let RNFS = require('react-native-fs');
let appSharedDir = '';
if(Platform.OS=='android'){
    appSharedDir = RNFS.DownloadDirectoryPath+'/MobScanner';
}

class MainApp extends Component{
    constructor(props){
        super(props);
        this.permissionsAndroid = this.permissionsAndroid.bind(this);
        this.getPermissions = this.getPermissions.bind(this);
        this.permissionsIos = this.permissionsIos.bind(this);
        this.appSharedFolder = this.appSharedFolder.bind(this);
    }

    permissionsAndroid(statuses){
        console.log(statuses)
        if(statuses[PERMISSIONS.ANDROID.CAMERA]=='denied'){
            const rationale = {
                title : 'Camera Permissions',
                message : 'Please Grant Camera Permissions to be able to Scan',
                buttonPositive : 'Allow',
                buttonNegative : 'Deny',
            };
            request(
                PERMISSIONS.ANDROID.CAMERA,
                rationale).then(result=>console.log(result));
        }
        if(statuses[PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE]=='denied'){
            const rationale = { 
                title : 'Camera Permissions',
                message : 'Please Grant Camera Permissions to be able to Scan',
                buttonPositive : 'Allow',
                buttonNegative : 'Deny',
            };  
            request(
                PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE,
                rationale).then(result=>console.log(result));
        }
        if(statuses[PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE]=='denied'){
            const rationale = { 
                title : 'Camera Permissions',
                message : 'Please Grant Camera Permissions to be able to Scan',
                buttonPositive : 'Allow',
                buttonNegative : 'Deny',
            };      
            console.log('asking');
            request(
                PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE,
                rationale).then(result=>console.log(result));
        }  
    }

    permissionsIos(statuses){
        if(statuses[PERMISSIONS.IOS.CAMERA]=='denied'){
            const rationale = {
                title : 'Camera Permissions',
                message : 'Please Grant Camera Permissions to be able to Scan',
                buttonPositive : 'Allow',
                buttonNegative : 'Deny',
            };
            request(
                PERMISSIONS.ANDROID.CAMERA,
                rationale).then(result=>console.log(result));
        }
        if(statuses[PERMISSIONS.IOS.MEDIA_LIBRARY]=='denied'){
            const rationale = {
                title : 'Camera Permissions',
                message : 'Please Grant Camera Permissions to be able to Scan',
                buttonPositive : 'Allow',
                buttonNegative : 'Deny',
            };
            console.log('asking');
            request(
                PERMISSIONS.IOS.MEDIA_LIBRARY,
                rationale).then(result=>console.log(result));
        }
    }
    
    async appSharedFolder(){
        const exists = await RNFS.exists(appSharedDir);
        console.log(exists);
        if(!exists){
            await RNFS.mkdir(appSharedDir);
        } 
    }
    
    componentDidMount(){
        this.getPermissions();
        this.appSharedFolder();
    }

    getPermissions(){
        if(Platform.OS == 'android'){
            checkMultiple([
                PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE,
                PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE,
                PERMISSIONS.ANDROID.CAMERA,
            ]).then(statuses=>this.permissionsAndroid(statuses));
        }
        else if(Platform.OS=='ios'){
            checkMultiple([
                PERMISSIONS.IOS.CAMERA,
                PERMISSIONS.IOS.MEDIA_LIBRARY
            ]).then(statuses=>this.permissionsIos(statuses));
        }
    }

    render(){
        return(
            <NavigationContainer ref = {navigatorRef}>
                <MainStack.Navigator 
                    initialRouteName = {'Scanner'}
                    options = {{swipeEnabled : false,}}>
                    <MainStack.Screen 
                        name = {'Scanner'}
                        component = {ScanContainer}/>
                </MainStack.Navigator>
            </NavigationContainer>
        );
    }
}

export default MainApp;
