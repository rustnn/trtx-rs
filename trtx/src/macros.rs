// Keep feature availability and its docs.rs badge in sync.
// The predicate supports the same syntax as #[cfg], including all/any/not.
macro_rules! cfg_feature {
    ($cfg:meta => { unsafe extern $abi:literal { $($item:item)* } }) => {
        unsafe extern $abi {
            $(
                #[cfg($cfg)]
                #[cfg_attr(docsrs_trtx, doc(cfg($cfg)))]
                $item
            )*
        }
    };
    ($cfg:meta => { $($item:item)* }) => {
        $(
            #[cfg($cfg)]
            #[cfg_attr(docsrs, doc(cfg($cfg)))]
            $item
        )*
    };
}
