from __future__ import annotations

import streamlit as st

from amc.pdac_classifier import PDACRecapper


class PDACAnalysis(PDACRecapper):
    @classmethod
    @st.cache_resource
    def get_instance(cls) -> PDACAnalysis:
        return cls()

    def __init__(self):
        super().__init__()

    def page_home(self) -> None:
        st.title("CT 판독문 유형 분류")

        page = st.sidebar.selectbox(
            "페이지 선택",
            ("Statistics", "Deviates", "Difference")
        )

        st.sidebar.selectbox()

        if page == "Statistics":
            self.page_statistics()

        elif page == "Deviates":
            self.page_deviates()

        else:
            self.page_difference()

    def page_statistics(self) -> None:
        st.write("CT 판독문 유형 분류기 결과 통계")
        pass

    def page_deviates(self) -> None:
        st.write("CT 판독문 유형 분류기 오분류 샘플")
        pass

    def page_difference(self) -> None:
        st.write("CT 판독문 유형 분류기 결과 차이")
        pass

if __name__ == "__main__":
    pdac_analysis = PDACAnalysis.get_instance()

    pdac_analysis.page_home()
