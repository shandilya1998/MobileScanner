import React, {Component} from 'react';
import {Platform} from 'react-native';
import {createDrawerNavigator } from '@react-navigation/drawer';
import {NavigationContainer} from '@react-navigation/native';
import {navigatorRef} from './rootNavigator';
import MobileScanner from './containers/MobileScanner';
import ScanContainer from './containers/ScanContainer';
import {checkMultiple, 
        PERMISSIONS,
        requestMultiple} from 'react-native-permissions';

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
        const permissions = new Array();
        if(statuses[PERMISSIONS.ANDROID.CAMERA]=='denied'){
            permissions.push(PERMISSIONS.ANDROID.CAMERA);
        }
        if(statuses[PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE]=='denied'){
            permissions.push(PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE);
        }
        if(statuses[PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE]=='denied'){
            permissions.push(PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE);
        } 
        requestMultiple(permissions).then(
            (statuses) => {
                console.log(statuses[PERMISSIONS.ANDROID.CAMERA]);
                console.log(statuses[PERMISSIONS.ANDROID.WRITE_EXTERNAL_STORAGE]);
                console.log(statuses[PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE]);
            }
        ); 
    }

    permissionsIos(statuses){
        const permissions = new Array();
        if(statuses[PERMISSIONS.IOS.CAMERA]=='denied'){
            permissions.push(PERMISSIONS.IOS.CAMERA);
        }
        if(statuses[PERMISSIONS.IOS.MEDIA_LIBRARY]=='denied'){
            permissions.push(PERMISSIONS.IOS.MEDIA_LIBRARY);
        }
        requestMultiple(permissions).then(
            (statuses) => {
                console.log(statuses[PERMISSIONS.IOS.CAMERA]);
                console.log(statuses[PERMISSIONS.IOS.MEDIA_LIBRARY]);
            }   
        ); 
        
    }
    
    async appSharedFolder(){
        const exists = await RNFS.exists(appSharedDir);
        console.log(exists);
        if(!exists){
            console.log('testt');
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
