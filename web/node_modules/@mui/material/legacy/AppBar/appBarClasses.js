import { unstable_generateUtilityClasses as generateUtilityClasses } from '@mui/utils';
import generateUtilityClass from '../generateUtilityClass';
export function getAppBarUtilityClass(slot) {
  return generateUtilityClass('MuiAppBar', slot);
}
var appBarClasses = generateUtilityClasses('MuiAppBar', ['root', 'positionFixed', 'positionAbsolute', 'positionSticky', 'positionStatic', 'positionRelative', 'colorDefault', 'colorPrimary', 'colorSecondary', 'colorInherit', 'colorTransparent', 'colorError', 'colorInfo', 'colorSuccess', 'colorWarning']);
export default appBarClasses;