@java.lang.Override
public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
        return true;
    }
    if (!(obj instanceof protobuf.http.UserGroupProto.GetTokenS)) {
        return super.equals(obj);
    }
    protobuf.http.UserGroupProto.GetTokenS other = (protobuf.http.UserGroupProto.GetTokenS) obj;
    boolean result = true;
    result = result && getHOpCode().equals(other.getHOpCode());
    result = result && getTokenId().equals(other.getTokenId());
    result = result && getTokenExpireTime().equals(other.getTokenExpireTime());
    return result;
}